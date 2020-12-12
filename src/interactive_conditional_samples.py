#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
from google_trans_new import google_translator
import csv
import random

import model, sample, encoder

def translate(items):
    translator = google_translator()
    if type(items) == "list":
        ret = []        
        for item in items:
            ret.append(translator.translate(item, lang_tgt = 'hu'))
        return ret
    else:
        return translator.translate(items, lang_tgt = 'hu')
def selectRandom (items, minm, maxm):
    count = random.randint(minm, maxm)
    return random.sample(items, count)

def addImages(txt, imgs):
    try:
        ll = txt.split("\n")
        img = random.choice(imgs)
        img2 = random.choice(imgs)
        cnt1 = (len(ll) // 2) //2
        cnt2 = cnt1 + (len(ll) // 2)    
        out = "\n".join(ll[0:cnt1])
        out = out + " <img src=" + img + ">"
        out = out + "\n".join(ll[cnt1:cnt2])
        out = out + " <img src=" + img2 + ">"
        out = out + "\n".join(ll[cnt2:])
        return out
    except Exception as e:
        print(e)
        return txt

    
def interact_model(
    #file1,file2,file3,
    model_name='1558M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=0.8,
    top_k=40,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    #print(file1)
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))


    if length is None:
        length = 1000#hparams.n_ctx - 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)
        try:
            os.remove("output.csv")
        except:
            pass
        outpt = csv.writer(open('output.csv', 'w',  encoding='utf-8'))
        outpt.writerow(["keyword", "GUID", "Description", "Tags", "Article", "Category"])
        # open title file
        with open('titles.txt') as f1:
            titles = f1.readlines()

        # open keywords file
        with open('keywords.txt') as f2:
            keywords = f2.readlines()

        # open title file
        with open('images.txt') as f3:
            images = f3.readlines()

        for i, title in enumerate (titles):  
            print("=" * 20)  
            print("Generating text for: ", title)               
            print("=" * 20)
            context_tokens = enc.encode(title)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    article = enc.decode(out[i])
                    #print(article)
                    #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    #print(article)
                    #print("=" * 60)
                    #translator = google_translator()
                    #print(translator.translate(article, lang_tgt = 'hu')) 
            
            title = translate(title)
            keyword = translate(keywords[i % len(keywords)])
            #print(article)            
            article = translate(article)
            tags = translate(",".join(selectRandom(keywords,3,4)))
            categories = translate(",".join(selectRandom(keywords,1,2)))
            article = addImages(article,images)
            outpt.writerow([keyword, i+1, title, tags, article, categories])

            #print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)    
    #interact_model()

