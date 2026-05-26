# End-to-End Medical-Chatbot using gpt-4o

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create an environment after opening the repository

```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```




```
# run the following command
python store_index.py
```

```
# Finally run the following command
python app.py