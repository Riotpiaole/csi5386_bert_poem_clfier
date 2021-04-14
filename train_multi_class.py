import torch 
from pdb import set_trace
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler , random_split , DataLoader
from transformers import get_linear_schedule_with_warmup
from multi_cls_bert import BertForMultiLabelSequenceClassification
from train import evaluation_metric_res , evaluation , format_time
from collections import defaultdict
from os import mkdir
import time 
from os.path import exists
from transformers import AdamW, BertConfig
import pandas as pd

TOP_CATEGORIES = 17

BATCH_SIZE = 12
EPOCHS = 30
dataset = pickle.load(
    open("./poem_clf_dataset/dataset.pkl", "rb"))

TRAIN_SIZE = int(0.8 * len(dataset))
VALIDATION_SIZE = int(0.1 * len(dataset))
TEST_SIZE = int(0.1 * len(dataset)) + 1

train_dataset , val_dataset, test_dataset = random_split(
    dataset, [TRAIN_SIZE, VALIDATION_SIZE , TEST_SIZE])

train_dataloader , validation_dataloader, test_dataloader = \
    DataLoader(
        train_dataset, 
        sampler = RandomSampler(train_dataset), # Select batches randomly
        batch_size = BATCH_SIZE), \
    DataLoader(val_dataset, 
        sampler = RandomSampler(val_dataset), # Select batches randomly
        batch_size = BATCH_SIZE), \
    DataLoader(test_dataset,
        sampler = RandomSampler(test_dataset), # Select batches randomly
        batch_size = BATCH_SIZE), \

def train_multi_label(model, epochs, device= torch.device("cpu"), LABEL="multi_class_poem"):
    model.cuda()
    total_t0 = time.time()
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
            
            loss , logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            model.zero_grad()        

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
        evaluation(validation_dataloader, model,"Validation", training_stats, device=device)

    
        
        if epoch_i >= 0 or epoch_i % 15 == 0:
            
            if not exists(f"./trained_weights/{epoch_i}"): mkdir(f"./trained_weights/{epoch_i}")
            if not exists(f"./result/{epoch_i}"): mkdir(f"./result/{epoch_i}")

            df = pd.DataFrame(training_stats)
            df.to_csv(f"./result/{epoch_i}/{LABEL}_epoch_{epoch_i}_bert_training_result.csv", index=False) 
            torch.save(model.state_dict(), f"./trained_weights/{epoch_i}/{LABEL}_epoch_{epoch_i}_bert_model.pth")
        
            test_stats = defaultdict(lambda : [])
            
            print("")
            print("Running Testing...")
            evaluation(test_dataloader, model,"Test", test_stats, device=device)
            test_stats = pd.DataFrame(test_stats)
            test_stats.to_csv(f"./result/{epochs}/{LABEL}_epoch_{epochs}_testing_result.csv", index=False)
        
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    


if __name__ == "__main__":
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
    
    model = BertForMultiLabelSequenceClassification.from_pretrained( 
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 17, # The number of output labels--2 for binary classification.   
    )
    train_multi_label(model, 1, device)