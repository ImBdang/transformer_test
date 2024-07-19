import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
#bdang
from datasets import load_dataset
from dataset import BilingalDataset, causal_mask

from model import build
from config import get_config, get_weights_file_path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from tqdm import tqdm

import warnings


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    #precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    #khoi tao decoder input voi SOS
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        #tao mask cho decoder
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        #tinh toan output
        out = model.decoder(encoder_output,source_mask, decoder_input, decoder_mask)
        #lay token tiep theo 
        prob = model.linearMethod(out[:, -1])
        #chon token voi ti le cao nhat
        _, next_word = torch.max(prob, dimt=1) # _, o day thuc chat co 2 gia tri la value va index, viec dung dau gach duoi de bo qua cac gia tri khong can thiet trong python
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model,   validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            #print ra console
            print_msg('-'*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            

#cho nay tao tokenizer
def get_all_sentences(ds, lan):
    for item in ds:
        yield item["translation"][lan]

def get_or_build_tokenizer(config, ds, lan):
    tokenizer_path = Path(config["tokenizer_file"].format(lan))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_token = ["[UKN]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lan), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("opus_books", f"{config['lan_src']}-{config['lan_tgt']}", split="train")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lan_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lan_tgt"])

    #lay 90% train, 10% cho validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingalDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lan_src"], config["lan_tgt"], config["seq_len"])
    val_ds = BilingalDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lan_src"], config["lan_tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0
    for i in ds_raw:
        src_ids = tokenizer_src.encode(i["translation"][config["lan_src"]]).ids
        tgt_ids = tokenizer_src.encode(i["translation"][config["lan_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Do dai cau lon nhat cua src: {max_len_src}")
    print(f"Do dai cau lon nhat cua tgt: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang su dung {device}")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #tensor board
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    init_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Dang load model {model_filename}")
        state = torch.load(model_filename)
        init_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loass_func = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    for epoch in range(init_epoch, config["num_epochs"]):

        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Dang xu li epoch {epoch:02d}")
        for batch in batch_iterator:
            # model.train()
            encoder_input = batch["encoder_input"].to(device) #(B, seq)
            decoder_input = batch["decoder_input"].to(device) #(B, seq)
            encoder_mask = batch["encoder_mask"].to(device) #(B, 1, 1, seq)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, seq, seq)

            #run tensor through transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            linear_output = model.linearMethod(decoder_output) #(B, seq, tgt_vocab_size)

            label = batch["label"].to(device) #(B, seq)
            loss = loass_func(linear_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) #(B, seq, tgt_vocab_size) -> (B * seq, tgt_vocab_size)
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            #truy cap loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            #lan truyen nguoc
            loss.backward()

            #update weights
            optimizer.step()
            optimizer.zero_grad()

            #run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            global_step += 1

        #luu model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
