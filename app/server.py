##################################### import dependencies #############################################

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from tqdm import trange
import os
import spacy
import ftfy
import zipfile
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import sys
from pathlib import Path
import psutil
import gc
from question_prompts import answer_to_life, answer_prompt

##################################### set up app files #################################################

# set path relative to this python serve script
path = Path(__file__).parent

# create app
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

# set maximum # of tokens to generate
metadata = {"num_words": 50, "device": "cpu"}

##################################### download models ###################################################
                     
# link to zipped folders of fine-tuned gpt-2 (117M, small) models

## tip: if you host model on Google Drive or dropbox, use this link to turn share link into immediate download link:
## https://rawdownload.now.sh/

model_seneca_url = 'https://www.dropbox.com/sh/f2dnufzf5hfgj8z/AABV-tA7lgEaXTp_6NuI6Ukza?raw=1'
model_seneca_folder = 'output_seneca'

model_paul_graham_url = 'https://www.dropbox.com/sh/74vttsmmjz1skk5/AAAVquYXmXhSKlwBSPYHHVKxa?raw=1'
model_paul_graham_folder = 'output_paul_graham'

model_david_goggins_url = 'https://www.dropbox.com/sh/4xc04guuva3p0p8/AAC0OIYfgg-oxAEAyQeN2lyka?raw=1'
model_david_goggins_folder = 'output_david_goggins'

model_brene_brown_url = 'https://www.dropbox.com/sh/e2ld6q1bw4ucvk5/AACO4Ov_22HoCrfdSE9jV7oZa?raw=1'
model_brene_brown_folder = 'output_brene_brown'

model_tony_robbins_url = 'https://www.dropbox.com/sh/9eyh0xuz0xkrsuv/AAADxopggVkSdCteVM9APagPa?raw=1'
model_tony_robbins_folder = 'output_tony_robbins'

# download model function
async def download_model(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=None) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

# download model folders, unzip folders, remove zipped model files
async def setup_models():
    await download_model(model_tony_robbins_url, path/f'{model_tony_robbins_folder}.zip')
    with zipfile.ZipFile(path/f'{model_tony_robbins_folder}.zip', 'r') as zip_tr:
        zip_tr.extractall(path/f'{model_tony_robbins_folder}')
    os.remove(path/f'{model_tony_robbins_folder}.zip')
    
    await download_model(model_brene_brown_url, path/f'{model_brene_brown_folder}.zip')
    with zipfile.ZipFile(path/f'{model_brene_brown_folder}.zip', 'r') as zip_bb:
        zip_bb.extractall(path/f'{model_brene_brown_folder}')
    os.remove(path/f'{model_brene_brown_folder}.zip')

    await download_model(model_seneca_url, path/f'{model_seneca_folder}.zip')
    with zipfile.ZipFile(path/f'{model_seneca_folder}.zip', 'r') as zip_sen:
        zip_sen.extractall(path/f'{model_seneca_folder}')
    os.remove(path/f'{model_seneca_folder}.zip')

    await download_model(model_paul_graham_url, path/f'{model_paul_graham_folder}.zip')
    with zipfile.ZipFile(path/f'{model_paul_graham_folder}.zip', 'r') as zip_pg:
        zip_pg.extractall(path/f'{model_paul_graham_folder}')
    os.remove(path/f'{model_paul_graham_folder}.zip')

    await download_model(model_david_goggins_url, path/f'{model_david_goggins_folder}.zip')
    with zipfile.ZipFile(path/f'{model_david_goggins_folder}.zip', 'r') as zip_dg:
        zip_dg.extractall(path/f'{model_david_goggins_folder}')
    os.remove(path/f'{model_david_goggins_folder}.zip')

# download and set up models
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_models())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

##################################### calculate RAM usage ##########################################################

# used to test amount of RAM used at various stages of loading and running app: 
def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    ram_bytes = process.memory_info().rss # in bytes
    ram_mb = ram_bytes//1000000 # in megabytes
    ram_gb = round(ram_bytes/1000000000, 2) # in gigabytes
    return ram_gb, ram_mb

##################################### set gpt-2 language model functions ############################################

# functions from Hugging Face implementation of gpt-2: https://github.com/huggingface/transformers/ 
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def init(model_path, metadata):
    model.to(metadata["device"])

def sample_sequence(
    model,
    length,
    context,
    num_samples=1,
    temperature=1,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    device="cpu",
):
    context = torch.tensor(context, dtype=torch.long, device="cpu")
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {"input_ids": generated}
            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.0)

            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated.view(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

##################################### set app's prediction function #####################################

def predict(sample, metadata, model, tokenizer):
    
    indexed_tokens = tokenizer.encode(sample)
    output = sample_sequence(
        model, metadata.get("num_words"), indexed_tokens, device=metadata["device"]
    )
    return tokenizer.decode(
        output[0, 0:].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
    )

##################################### show html homepage ###############################################

@app.route('/')
def index(request):
    html = path/'index.html'
    return HTMLResponse(html.open().read())

###################################### load model and generate language ################################

@app.route('/analyze', methods=['POST'])
async def analyze(request):

    # identify mentor and question that user selected
    input_form = await request.form()
    question_prompt = input_form['question'] # used to identify the language model prompt category
    mentor = input_form['mentor'] # used for model generation signature
    
    # format linked text
    hyperlink_format = '<a href="{website}" style="color:blue; border-bottom: 1px solid" target="_blank" rel="noopener">{text}</a>'

    # generate answer prompts for model
    if question_prompt == "42":
        answer = answer_to_life()
        return JSONResponse({'result': answer})
    elif question_prompt == "empty":
        anwser_link = hyperlink_format.format(website='https://giphy.com/gifs/life-bid-dJv3R2vXEjPRm/fullscreen', text="I'll try to answer.")
        answer = "Choose your question and "+anwser_link
        return JSONResponse({'result': answer})
    else:
        answer = answer_prompt(question_prompt) # use a language model prompt/seed from "question_prompts.py"

    # load gpt-2 model for selected mentor
    if mentor == "seneca":
        model = GPT2LMHeadModel.from_pretrained(path/f'{model_seneca_folder}')
        tokenizer = GPT2Tokenizer.from_pretrained(path/f'{model_seneca_folder}')
    elif mentor == "paul-graham":
        model = GPT2LMHeadModel.from_pretrained(path/f'{model_paul_graham_folder}')
        tokenizer = GPT2Tokenizer.from_pretrained(path/f'{model_paul_graham_folder}')
    elif mentor == "david-goggins":
        model = GPT2LMHeadModel.from_pretrained(path/f'{model_david_goggins_folder}')
        tokenizer = GPT2Tokenizer.from_pretrained(path/f'{model_david_goggins_folder}')
    elif mentor == "brene-brown":
        model = GPT2LMHeadModel.from_pretrained(path/f'{model_brene_brown_folder}')
        tokenizer = GPT2Tokenizer.from_pretrained(path/f'{model_brene_brown_folder}')
    elif mentor == "tony-robbins":
        model = GPT2LMHeadModel.from_pretrained(path/f'{model_tony_robbins_folder}')
        tokenizer = GPT2Tokenizer.from_pretrained(path/f'{model_tony_robbins_folder}')

    # generate model prediction (string of text)
    model_prediction = predict(answer, metadata, model=model, tokenizer=tokenizer)
    ram_mb = memory_usage_psutil()[1]
    print(f"After generating prediction, {ram_mb} MB of RAM is in use")
    
    # reset variables to save RAM
    model = None
    tokenizer = None

    # garbage collect to free up memory
    gc.collect()
    ram_mb = memory_usage_psutil()[1]
    print(f"After garbage collect, {ram_mb} MB of RAM is in use")

    # find position of final punctuation mark in model prediction (period, question, or exclamation)
    punct_period = model_prediction.rfind('.')
    punct_excl = model_prediction.rfind('!')
    punct_quest = model_prediction.rfind('?')

    punct_per_quote = model_prediction.rfind('."')+1 # if ends in quotation, include final quotation
    punct_exc_quote = model_prediction.rfind('!"')+1 # if ends in quotation, include final quotation
    punct_ques_quote = model_prediction.rfind('?"')+1 # if ends in quotation, include final quotation
    
    # find position to truncate model prediction
    min_chars = 100 # used as default in case generated language does not include ending punctuation and returns '-1'  
    punct_max = max(punct_period, punct_excl, punct_quest, punct_per_quote, punct_exc_quote, punct_ques_quote, min_chars)
    
    # truncate model prediction so it ends with punctuation mark
    trunc_model_prediction = model_prediction[:punct_max+1]
    print(trunc_model_prediction) # printed in logs to ensure it is working as expected

    # add a closing signature (not generated) for mentor
    if mentor == "seneca":
        sig_start = "</br></br>Get the rest of my free stoic letters "
        sig_link = hyperlink_format.format(website='https://tim.blog/2017/07/06/tao-of-seneca/', text='here')
        sig_end = ". Farewell."
        sig = sig_start + sig_link + sig_end
    elif mentor == "paul-graham":
        sig_start = "</br></br>Want to start a startup? Get funded by "
        sig_link = hyperlink_format.format(website='https://www.ycombinator.com/apply.html', text='Y Combinator')
        sig = sig_start + sig_link
    elif mentor == "brene-brown":
        sig_start = "</br></br>Be vulnerable. Be seen. "
        sig_link = hyperlink_format.format(website='https://youtu.be/-s6DQrqVHxM?t=37', text='Get in the arena.')
        sig = sig_start + sig_link
    elif mentor == "david-goggins":
        sig_start = "</br></br>"
        sig_link = hyperlink_format.format(website='https://www.youtube.com/watch?v=DS0ed93UQeY', text='#TakingSouls')
        sig = sig_start + sig_link
    elif mentor == "tony-robbins":
        sig_start = "</br></br>God Bless. Thank You.</br>#IamNotYourGuru"
        sig = sig_start
    
    return JSONResponse({'result': trunc_model_prediction+sig})

##################################### calculate RAM use to serve ###########################################

ram_mb = memory_usage_psutil()[1]
print(f"After starting serving process (before loading model), {ram_mb} MB of RAM is in use")

##################################### run app ##############################################################

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=8080)

