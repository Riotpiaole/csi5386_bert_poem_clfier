import torch
import pandas as pd 
from os import mkdir
from pdb import set_trace
from os.path import exists
from collections import defaultdict

from bert import obtain_dataset_by_label, train_df
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig

epochs = 30
LABEL = 'social'

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

import numpy as np
from sklearn.metrics import recall_score , precision_score , f1_score
# Function to calculate the accuracy of our predictions vs labels
def evaluation_metric_res(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    recall_res_score = recall_score( labels_flat , pred_flat )
    precision_res_score = precision_score( labels_flat , pred_flat )
    f1_res_score = f1_score( labels_flat , pred_flat )

    return (np.sum(pred_flat == labels_flat) / len(labels_flat), 
            recall_res_score,
            precision_res_score, 
            f1_res_score)

def evaluation( data_loader, model , tag="val", next_stats= {}, save_res=False):
    dfs = []
    t0 = time.time()

    total_eval_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in data_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            res = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            loss, logits = res['loss'], res['logits']

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        eval_accuracy, eval_recall , eval_precision , eval_f1_score = \
            evaluation_metric_res(
                logits, label_ids)
        
        total_eval_accuracy += eval_accuracy 
        total_f1_score += eval_f1_score
        total_precision += eval_precision
        total_recall += eval_recall

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(data_loader)   
    avg_val_precision = total_precision / len(data_loader)
    avg_val_f1score = total_f1_score / len(data_loader)
    avg_val_recall = total_recall / len(data_loader)

    print(f"  {tag}"+" Accuracy: {:.2f}".format(avg_val_accuracy))
    print(f"  {tag}"+" Recall: {:.2f}".format(avg_val_recall))
    print(f"  {tag}"+" Precision: {:.2f}".format(avg_val_precision))
    print(f"  {tag}"+" F1Score: {:.2f}".format(avg_val_f1score))  
    
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(data_loader)
    
    # Measure how long the validation run took.
    validation_time_ps = time.time() - t0
    validation_time = format_time(validation_time_ps)
    
    print(f"  {tag}"+" Loss: {0:.2f}".format(avg_val_loss))
    print(f"  {tag} took: " + validation_time)

    # Record all statistics from this epoch.
    
    next_stats[f'{tag}_Loss'].append(avg_val_loss)
    next_stats[f'{tag}_Accuracy'].append(avg_val_accuracy)
    next_stats[f'{tag}_Recall'].append(avg_val_recall)
    next_stats[f'{tag}_Precision'].append(avg_val_precision)
    next_stats[f'{tag}_F1Score'].append(avg_val_f1score)
    next_stats[f'{tag}_Time'].append(validation_time_ps)
    
    

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.

# Measure the total training time for the whole run.
total_t0 = time.time()


import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def train(model, epochs, label=LABEL, device = device):      
    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.

    train_dataloader , validation_dataloader, test_dataloader, _ = obtain_dataset_by_label( label, train_df=train_df )
   

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )



    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    
    training_stats = defaultdict(lambda : [] )
    # For each epoch...
    if not exists(f"./result/{epochs}"):
        mkdir(f"./result/{epochs}")

    for epoch_i in range( epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 1 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed), end="\r")

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            res = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
            loss , logits = res['loss'] , res['logits']

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time_ps = time.time() - t0
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        training_stats['Training_Loss'].append( avg_train_loss)
        training_stats['Training_Time'].append( training_time_ps)
        training_stats['Training_Time_format'].append( training_time_ps)
        
        # ========================================
        #          Test & Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")
        evaluation(validation_dataloader, model,"Validation", training_stats)

    
        
        if epoch_i > 0 and epoch_i % 15 == 0:
            
            if not exists(f"./trained_weights/{epoch_i}"): mkdir(f"./trained_weights/{epoch_i}")
            if not exists(f"./result/{epoch_i}"): mkdir(f"./result/{epoch_i}")

            df = pd.DataFrame(training_stats)
            df.to_csv(f"./result/{epoch_i}/{LABEL}_epoch_{epoch_i}_bert_training_result.csv", index=False) 
            torch.save(model.state_dict(), f"./trained_weights/{epoch_i}/{LABEL}_epoch_{epoch_i}_bert_model.pth")
        
            test_stats = defaultdict(lambda : [])
            
            print("")
            print("Running Testing...")
            evaluation(test_dataloader, model,"Test", test_stats)
            test_stats = pd.DataFrame(test_stats)
            test_stats.to_csv(f"./result/{epochs}/{LABEL}_epoch_{epochs}_testing_result.csv", index=False)
        
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    

if __name__ == "__main__":
    # train(model, 150, 'agency')
    train(model, 30, 'social')

    time.sleep(150)    
    import os 
    os.system("shutdown /s /t 1") 


