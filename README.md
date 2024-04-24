## 💽 Dataset 
XTREME is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks.

## 📚 Approach 
1. Get data and properly create text and label (Can be done using https://explosion.ai/demos/displacy-ent.
2. Use transformer Roberta architecture for training the ner tagger.
3. Use hugging face for Robereta Tokenizer.
4. Train and Deploy model for use-cases.

## 🚀 API 
<img width="790" alt="image" src="https://user-images.githubusercontent.com/57321948/200125543-da8286e2-dda6-4670-bf40-0c1584f97a60.png">

## 🧑‍💻 How to setup

create fresh conda environment 
```python
conda create -p ./env python=3.7 -y
```
activate conda environment
```python
conda activate ./env
```
Install requirements
```python
pip install -r requirements.txt
```

To run inferencing
```python
python app.py
```

To launch swagger ui
```python
http://localhost:8080/docs
```
## 🧑‍💻 Tech Used
1. Natural Language processing
2. Pytorch 
3. Transformer 
4. FastApi 

## Deployment
1. AWS ECR
2. AWS EC2

## 🏭 Industrial Use-cases 
1. Search and Recommendation system 
2. Content Classification 
3. Customer Support 
4. Research Paper Screening 
5. Automatically Summarizing Resumes 

## 👋 Conclusion 
In this project, we have demonstrated how to train a custom NER model using the XTREME dataset and the Roberta transformer architecture. We used the Hugging Face library to easily load and preprocess the data and train the model. We also deployed the model using FastAPI and Docker, making it easy to use in various applications.

This solution can be used in various industrial applications such as search and recommendation systems, content classification, customer support, and research paper screening.

Overall, this project provides a simple and effective solution for training and deploying custom NER models, making it easier and faster to extract important information from unstructured text data.
