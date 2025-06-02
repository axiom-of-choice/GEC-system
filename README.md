# What is this for?

This is a gramatical error correction (GEC) system (it lacks of a serving layer -API- though) to correct english written texts.

# Table of contents:
- [What is this for?](#what-is-this-for)
- [Structure of the project](#structure-of-the-project)
    - [Project Structure](#project-structure)
- [How did we do it?](#how-did-we-do-it)
    - [Preparing the dataset](#preparing-the-dataset)
        - [FCE data](#fce-data)
        - [Medical data](#medical-data)
    - [Modelling](#modelling)
        - [Training the T5 model](#training-the-t5-model)
        - [Prompt engineering for LLama3 model](#prompt-engineering-for-llama3-model)
    - [Evaluation](#evaluation)
        - [T5](#t5)
        - [LLAMA](#llama)
    - [How can we improve?](#how-can-we-improve)
        - [T5](#t5-1)
        - [LLAMA](#llama-1)
    - [Analyzing performance](#analyzing-performance)
        - [T5](#t5-2)
- [Conclusion](#conclusion)

# Structure of the project
## Project Structure

```
├── README.md
│
├── config # 
│   ├── constants.py   # File containing cosntants
│   ├── prompt_general.txt   # Prompt to be used on llama3 for general purpose corrrections
│   └── prompt_medical.txt   # Prompt to be used on llama3 for medical corrections
│
├── data # Directory containing data
│
├── gec
│   ├── dataset_loader.py   # M2 file downloader and parser
│   ├── evaluator.py   # Helper class to perform GLEU and exact match evaluation
│   ├── inference.py   # Helper class to define inference engines for both models
│   ├── preprocessor.py   # Helper class to run a preprocessing over the parser M2 files to finetune T5 model
│   └── trainer.py   # Helper class to train T5 model
│
├── notebooks
│   └── GEC.ipynb   # Full notebook 
│
├── scripts
│   ├── preprocess.py      # Script to parse and preprocess M2 files
│   ├── predictions_builder.py  # CLI script to run inference on the already preprocessed.
│   └── evaluation_runner.py     # CLI script to run evaluation using the inference datasets. 
│
├── models                 # Contains the finetuned models
│   ├── t5_final           #   
│   └── t5_medical_finetuned # 
└── static
    └── architecture.md    # Detailed description of project components and workflow
```

# How did we do it?

To achieve this task, we tested two models:
- T5 small

    I picked this model because is a general text-to-text transformer model, and it is small but good enough to use it in a first iteration.

    This model was finetuned using fce [BEA19 FCE V2.1 data](https://www.cl.cam.ac.uk/research/nl/bea2019st/).
     
    We had to parse M2 file format, and preprocess it using T5 Tokenizer, as you'll see in the notebook or the python files.

- Llama 3 8b
    I picked this LLM because it is open source, good enough and i could run it on local using [ollama API](https://ollama.com/library/llama3) for running the infernce.

    This model was not finetuned but prompt engineered to get the best results.

## Preparing the dataset:
When doing this step, we had to first explore the format of the data.
### FCE data
For the FCE data, the files have one or more line per sentence, and one or more line per annotation, grouped together by a blank line.
If there si a *noop* tag the there is no edit.
```
S Dear Sir or Madam ,
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0

S I am writing in order to express my disappointment about your musical show " Over the Rainbow " .
A 9 10|||R:PREP|||with|||REQUIRED|||-NONE-|||0

S I saws the show 's advertisement hanging up of a wall in London where I was spending my holiday with some friends . I convinced them to go there with me because I had heard good references about your Company and , above all , about the main star , Danny Brook .
A 1 2|||R:VERB:TENSE|||saw|||REQUIRED|||-NONE-|||0
A 8 9|||R:PREP|||on|||REQUIRED|||-NONE-|||0
A 36 37|||R:NOUN|||reviews|||REQUIRED|||-NONE-|||0
A 37 38|||R:PREP|||of|||REQUIRED|||-NONE-|||0
A 45 46|||R:PREP|||because of|||REQUIRED|||-NONE-|||0

S The problems started in the box office , where we asked for the discounts you announced in the advertisement , and the man who was selling the tickets said that they did n't exist .
A 3 4|||R:PREP|||at|||REQUIRED|||-NONE-|||0
```
Take for example the first example, there is no edit needed, the sentence is already correct.
```
S Dear Sir or Madam ,
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
```
But in the second one, theree is some edits, it replaces *about* -> *with*
```
S I am writing in order to express my disappointment about your musical show " Over the Rainbow " .
A 9 10|||R:PREP|||with|||REQUIRED|||-NONE-|||0
```
So when parsing the data both recors look like this:
```
Wrong:
Dear Sir or Madam ,
Corrected:
Dear Sir or Madam ,
Wrong:
I am writing in order to express my disappointment about your musical show " Over the Rainbow " .
Corrected:
I am writing in order to express my disappointment with your musical show " Over the Rainbow " .
```

We also had to add preporcessor of the data using T5 Tokenizer to actually train the model using the trai, dev and test data extracted from the T5. 

To do so we only had to chose which has the max_length of the input and it was simply taken analyzing the maximimum sentence length of all the data, because T5 small has a default of 512 that might be too short.

I also observed that for the test data, this were the top10 edits. (edit_type, ocurrences)
```
[('noop', 903),
 ('R:OTHER', 617),
 ('R:NOUN', 576),
 ('R:SPELL', 423),
 ('M:DET', 335),
 ('R:ORTH', 263),
 ('R:PREP', 254),
 ('R:VERB', 233),
 ('R:VERB:TENSE', 165),
 ('R:DET', 134)]
```
### Medical data
This data was actually pretty clean andit was a .csv format, with only two columns: wrong and correct. for each sentence

And since we did not trained T5 (at first) with that data, we did not have to preprocess the data (at this step)
This is how it looks like.

```
incorrect_sentence, correct_sentence
"The patient report severe dyspnea and bilateral lower extremity edema following administration of intravenous furosemide", "The patient reported severe dyspnea and bilateral lower extremity edema following administration of intravenous furosemide."
```

I observed though the sentences had a strong medical language, and the most common edits were: (edit_type, ocurrences)
```
[('R:VERB:TENSE', 177),
 ('M:DET', 36),
 ('R:VERB:FORM', 7),
 ('R:VERB:SVA', 6),
 ('R:PREP', 5),
 ('M:OTHER', 1),
 ('R:OTHER', 1),
 ('U:OTHER', 1),
 ('M:VERB:TENSE', 1),
 ('R:ADV', 1)]
```

## Modelling
### Training the T5 model:
Once we had the data preprocessed, we just created a T5Trainer helper class that will help us to orchestrate the training of the data. 

The model was trained using sopme default parameters and this other ones manually set
* epochs: 3 (Due to GPU limitations)
* saving checkpoints every epoch
* Early stopping with 3 patience and 0.0 threshold (It stops training after 3 iterations without improvement)
* Batch size: 8 (due to GPU limitations)
* Learning rate: 3e-4 (To test convergence)
* evaluation each 500 steps of loss for early stopping

The model lasted 2:25 to tain using a 16 GB of RAM, and a T4 GPU with 16GB of VRAM.

![alt text](/static/t5_general_finetune.jpg "Finetuning the model over FCE data")

The accuracy of the model later evaluated was good though for a first iteration.

### Prompt engineering for LLama3 model
We just passed some examples to the prompt and setting some parameters to the inference engine to achieve the task, such as:
* Temperature: 0.2 to avoid way too much "creativity"
* top_k: 20
* top_p: 0.5 To select the best answers
* seed: To get deterministic results

You can see the prompts in the /config/propmt_*.txt 

## Evaluation
To do the evaluation, we need to choose some metrics to perform the task. For this, we will do it using three baselines:

- Exact Match

Why do we use this metrics to evaluate GEC?
Exact match is pretty strict and shows how often the system produces a fully corrected sentence comparing with the test data. 100% deterministic so it is good for systems where only perfect corrections are allowed.
- Gleu

Why do we use it?
It is a sentence level variant of BLEU, it is less strict than exact match and reflecs better incremental improvements, not a 100% deterministic correction.


- ERRANT score
Why do we use it?
Because it is a standard metric for GEC, evaluates how well the system identifies and corrects specific errors. It provides a detailed edit level assement.
Basically it compare edits between the system and a reference m2 files (we had to create)

Note:

**The first two evaluations are done with the evaluator class defined below, using the inference data included in the repository (fce_predicted.csv for t5, fce_predicted_llama.csv for llama3). The data was generated using a script included in the repository (/scripts/predictions_builder), instructions about how to use the script are there.**

***The last score evaluation are done generating gold M2 files using an external framework, with the source data (incorrect), target (correct). We also had to generate M2 files for the predicted data using the source data(incorrect) and predicted data.***

```
!errant_parallel -orig ./data/medical_wrong.txt -cor ./data/medical_predicted_t5.txt -out ./data/m2/medical_predicted_t5.m2

!errant_compare -hyp ./data/m2/medical_predicted_t5.m2 -ref ./data/m2/medical_gold.m2
```
### T5 
* FCE test dataset
    The overall accuracy seems to be pretty solid for the FCE data having an average of 0.3833 exact match, and 0.7826 GLEU
    
    For ERRANT Score we achieved  the following results:
    ```
    =========== Span-Based Correction ============
    TP	    FP	    FN	    Prec	Rec	    F0.5
    1452	1411	3312	0.5072	0.3048	0.4477
    ==============================================
    ```
    Which is kinda solid for a baseline GEC system, since it basically has 0.45 of F0.5 score, and it's kinda conservative, since the precision is way higher than recall
    It actually was better over the "leave as it is" sentences, and those are most of the cases.

    Some examples of a correct and incorrect sentences and their accuracy could be:

    ```
    Source: When Alexander Graham Bell first invented the telephone, he had no idea of how popular it would be one day.
    Target: When Alexander Graham Bell first invented the telephone, he had no idea how popular it would be one day.
    Prediction: When Alexander Graham Bell first invented the telephone, he had no idea how popular it would be one day.

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------

    Source: Your sincerely,
    Target: Yours sincerely,
    Prediction: Yours sincerely,

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------
    Source: I will give you now some useful information.
    Target: I will now give you some useful information.
    Prediction: I will give you now some useful information.

    Exact Match: 0, GLEU: 0.5000
    ------------------------------------------------------------

    Source: But also, the museum have a lot of gardens with several statues.
    Target: But also the museum has a lot of gardens with several statues.
    Prediction: But also, the museum has a lot of gardens with several statues.

    Exact Match: 0, GLEU: 0.8333
    ------------------------------------------------------------
    ```
* Medical dataset
    Pretty poor perfomance due to lack of context and training over medical sentences wit an exact match of 0.13 and gleu of 0.78
    ERRANT score is around 
    ```
    =========== Span-Based Correction ============
    TP	FP	FN	Prec	Rec	    F0.5
    31	184	208	0.1442	0.1297	0.141
    ==============================================
    ```
    Most of the wrong corrections are related when adding prepositons "The" for example. 
    
    But it is good when correcting verb times.
    ```
    Source: Patient develop diabetic ketoacidosis with severe dehydration and electrolyte imbalances requiring intensive care management.
    Target: The patient developed diabetic ketoacidosis with severe dehydration and electrolyte imbalances requiring intensive care management.
    Prediction: Patient develops diabetic ketoacidosis with severe dehydration and electrolyte imbalances requiring intensive care management.

    Exact Match: 0, GLEU: 0.7778
    ------------------------------------------------------------

    Source: The radiation oncologist recommend intensity-modulated radiation therapy for locally advanced prostate adenocarcinoma.
    Target: The radiation oncologist recommended intensity-modulated radiation therapy for locally advanced prostate adenocarcinoma.
    Prediction: The radiation oncologist recommend intensity-modulated radiation therapy for locally advanced prostate adenocarcinoma.

    Exact Match: 0, GLEU: 0.7619
    ------------------------------------------------------------

    Source: Slit-lamp examination reveal anterior uveitis with keratic precipitates and posterior synechiae formation.
    Target: Slit-lamp examination revealed anterior uveitis with keratic precipitates and posterior synechiae formation.
    Prediction: Slit-lamp examination revealed anterior uveitis with keratic precipitates and posterior synechiae formation.

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------

    Source: She were admitted to the coronary care unit yesterday following acute ST-elevation myocardial infarction.
    Target: She was admitted to the coronary care unit yesterday following acute ST-elevation myocardial infarction.
    Prediction: She was admitted to the coronary care unit yesterday following acute ST-elevation myocardial infarction.

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------
    ```

### LLAMA
* Fce dataset:
    Fort his model, we have kinda poor perfomance on FCE data, since i did not improved the prompt good enough adding examples of corrections, and there's a lot of cases where the model overcorrects. 

    We have a 0.13 overall exact match and 0.60 GLUE, while the ERRANT score is

    ```
    =========== Span-Based Correction ============
    TP	    FP	    FN	    Prec	Rec	    F0.5
    1667	3509	3097	0.3221	0.3499	0.3273
    ==============================================
    ```

    It is baseline, but is not good. Some examples:
    ```
    Source: We have booked the hotel for the group.
    Target: We have booked the hotel for the group.
    Prediction: We have booked the hotel for the group.

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------

    Source: In the business world you use the phone all the time to speak with your clients, to make appointments, to order an article e. c. ; then, time is money!
    Target: In the business world you use the phone all the time to speak with your clients, to make appointments, to order an article etc. ; so, time is money!
    Prediction: In the business world, you use the phone all the time to speak with your clients, to make appointments, to order an article, etc. ; then, time is money!

    Exact Match: 0, GLEU: 0.7364
    ------------------------------------------------------------
    Source: It is going to be very lovely and enjoyable because I have lots of surprises for students.
    Target: It is going to be very lovely and enjoyable because I have lots of surprises for the students.
    Prediction: It is going to be very lovely and enjoyable because I have lots of surprises for the students.

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------

    ```

* Medical data:
    For this data, we have kinda good performance, since i was able to add some cases that captures the most frequent erross in the prompt.

    We have overall exact match of 0.504 and a GLEU score of 86%. For me, it seems good, since analyzing the outputs, the corrections are kindagood and the bad ones are not that bad.

    
    While errant score are the following numbers.

    ```
    =========== Span-Based Correction ============
    TP	FP	FN	Prec	Rec	    F0.5
    181	156	58	0.5371	0.7573	0.5703
    ==============================================
    ```

    We can clearly see how it suceeds on correcting the tense verb and sometimes exact match is not exact because it adds some punctuation (last example) but i guess the results are pertty good.

    ```
    Source: Fluorodeoxyglucose positron emission tomography show increased metabolic activity consistent with viable myocardium.
    Target: Fluorodeoxyglucose positron emission tomography showed increased metabolic activity consistent with viable myocardium.
    Prediction: Fluorodeoxyglucose positron emission tomography showed increased metabolic activity consistent with viable myocardium.

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------

    Source: The occupational therapist recommend adaptive equipment and environmental modifications for activities of daily living.
    Target: The occupational therapist recommended adaptive equipment and environmental modifications for activities of daily living.
    Prediction: The occupational therapist recommends adaptive equipment and environmental modifications for activities of daily living.

    Exact Match: 0, GLEU: 0.8000
    ------------------------------------------------------------

    Source: Magnetic resonance spectroscopy demonstrate reduced N-acetylaspartate levels consistent with neuronal loss and gliosis.
    Target: Magnetic resonance spectroscopy demonstrated reduced N-acetylaspartate levels consistent with neuronal loss and gliosis.
    Prediction: Magnetic resonance spectroscopy demonstrated reduced N-acetylaspartate levels consistent with neuronal loss and gliosis.

    Exact Match: 1, GLEU: 1.0000
    ------------------------------------------------------------

    Source: Immunohistochemistry staining show positive cytokeratin and negative vimentin consistent with epithelial tumor origin.
    Target: Immunohistochemistry staining showed positive cytokeratin and negative vimentin consistent with epithelial tumor origin.
    Prediction: Immunohistochemistry staining showed positive cytokeratin and negative vimentin, consistent with epithelial tumor origin.

    Exact Match: 0, GLEU: 0.7826
    ------------------------------------------------------------
    ```

## How can we improve?
### T5
* Picking a larger model, like a T5 base and giving it more time of training should improve the scores. Doing a better preprocess of the data because i did not run a broad analysis (could take way more time that i already invested doing this).
* As for the medical data, we need (and actually did it later) to finetune the model over the medical data as well so it can have enough context.
* Look for the type of corrections and examples that fails the most and add more of that in the data could also help.

### LLAMA

* For the FCE data, we need to pass more examples in the prompts that captures the whole behavior of the test data and run more tests (did not do it because inferencing is kinda expensive and takes a while, even doing it async and parallel  on my machine).

* For the medical dataset, again we can improve the prompt adding some cases that were not good captured by the first iteration and look for common patterns in the examples it fails the most to add them.

* For both cases finetuning the model (training it) over the data could also imoprove a lot the performance.


Of course i will need to enable an API to run the inference and serve it but i did not do it.

## Analyzing performance:
### T5
T5 model perfomance is kind agood when inferencing the data, it takes less than a second generating a response for a single record once the already trained model is loaded into the service.

**You can also run the prediction using batches in parallel and that speeds a lot inference time**

### LLAMA3
While llama3 is plug and play, the inference time (at least on my machine) takes around a second, and you cannot send batches of requests, you can only send them async that kinda speeds up the inference but still take its time.

# Conclusion:

T5 models are pretty good models doing the corrections if you train them in the correct way, using **a good ammount** of appropiately processed data.

Though it takes a while to train, and first versions of the systems needs some time to develop, the inference is pretty quick and in parallel if you enable it through an API, that's pretty advantageous on systems with a lot of traffic and where the response time is crucial.

It still has its limitations like max length of the input and output, and T5 model is not designed  for incremental (streaming) correction, they process the whole sentence at once. So it's better to have it for a batch processing or it will need some smart buffering approach (hybrid using RNN or incremental transformers)


On the other hand, LLAMA3 for example, has streaming capabilities, can handle long context, you don't even need to train it (you can use it vanilla and just do prompt engineering as we did), and if needed, can use just a small ammount of data to fine tune it. So it's kinda plug and play and could be useful in a first iteration.

But it has it's drawbacks. It does not support parallel request by default, so it could cause any problems on low-latency systems, you need huge compute resources to handle the model and deploy it.
