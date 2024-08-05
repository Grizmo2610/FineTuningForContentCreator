# Import necessary libraries
from google.generativeai import ChatSession
import google.generativeai as genai
from enum import Enum
import pandas as pd
import json
import os

# Path to your file that contains your API Google KEY
path = 'UserData/Keys/GeminiKey.key'
try:
    # Attempt to read the API key from the specified file and set it as an environment variable
    with open(path, 'r') as file:
        key = file.read().strip()
        os.environ["GEMINI_API_KEY"] = key
except FileNotFoundError:
    print(f"File '{path}' not found.")
except Exception as e:
    print(f"Error reading file '{path}': {e}")

# Set your default model for Gemini
GEMINI_DEFFAULT_MODEL = 'gemini-1.5-flash'
# Specify the language to be used (English or Vietnamese Only)
language = 'vietnamese'
# JSON system prompt template
JSON_SYSTEM_PROMPT = 'Bạn là AI được mô phỏng theo {creator} có nhiệm vụ là nội dung, kịch bản của một video dựa vào tiêu đề, mô tả của người dùng'

# Load the training prompt configuration from a JSON file
with open(f'Config/InputPrompt/{language}_train_prompt.json', encoding='utf-8') as f:
    train_prompt = json.load(f)

# Class for interacting with the Gemini model
class GeminiModel:
    def __init__(self, model = ..., key = ...) -> None:
        # Use the default model if none is provided
        if model == ...:
            model = GEMINI_DEFFAULT_MODEL
        self.model = genai.GenerativeModel(model)
        self.session = ChatSession(self.model)
        try:
            # Use the environment API key if none is provided
            if key == ...:
                self.key = os.environ["GEMINI_API_KEY"]
            else:
                self.key = key
        except:
            # Fallback key (replace with your actual key)
            self.key = 'YOUR-GEMINI-API-KEY'
        genai.configure(api_key=self.key)
    
    def set_key(self, key):
        # Set a new API key
        self.key = key
        genai.configure(api_key=self.key)
    
    def response(self, text, prompt):
        # Send a message to the Gemini model and retrieve the response
        response = self.session.send_message(f"{prompt}\n{text}\n")
        final = ''
        for chunk in response:
            final += chunk.text + '\n'
        return final.strip()

# Class for training the model with specific data
class TrainModel:
    def __init__(self, creator_name) -> None:
        self.creator_name = creator_name

    def _save_data(self, data: pd.DataFrame, encoding: str):
        # Create necessary directories if they do not exist
        try:
            os.makedirs('TrainedData')
        except:
            pass
        try:
            os.makedirs(f'TrainedData/{self.creator_name}')
        except:
            pass

        # Save the trained data as a CSV file
        path = f'TrainedData/{self.creator_name}/{self.creator_name}_trained_data.csv'
        data.to_csv(path, encoding=encoding, index=False)
        print(f'CSV file saved at {path}')

        # Prepare the data for JSONL format
        all_conversations = []
        for idx, row in data.iterrows():
            all_conversations.append({'messages':[
                {'role':'system', 'content': JSON_SYSTEM_PROMPT},
                {'role':'user', 'content': row['input_string']},
                {'role':'assistant', 'content': row['content']}
            ]})

        # Save the data as a JSONL file
        path = f'TrainedData/{self.creator_name}/{self.creator_name}_instances.jsonl'
        with open(path, 'w', encoding=encoding) as f:
            for conversation in all_conversations:
                json.dump(conversation, f, indent=4)
                f.write('\n')
        print(f'JSON data saved at {path}')

    def train(self, data, encoding: str = 'utf-8'):
        # Validate the input data type and length
        if isinstance(data, list) or isinstance(data, tuple) or isinstance(data, pd.Series):
            if len(data) < 1:
                raise ValueError('At least 1 scenario is required')
        else:
            raise TypeError('Invalid data type')

        # Define the columns for the training data
        COLUMNS = ['prompt', 'title', 'description', 'content']
        container = {x: [] for x in COLUMNS}
        train_model = GeminiModel()

        # Process each data point and generate responses using the model
        for i in range(len(data)):
            print(f'Processing {i + 1}/{len(data)}')
            container['prompt'].append(train_model.response(data[i], train_prompt['prompt']))
            container['title'].append(train_model.response(data[i], train_prompt['title']))
            container['description'].append(train_model.response(data[i], train_prompt['description']))
            container['content'].append(train_model.response(data[i], train_prompt['content']).replace('*', '').replace('#', '').replace('\n', '\\n').replace(',', '').strip())
        
        # Convert the container to a DataFrame
        data = pd.DataFrame(container)
        data['content'] = data['content'].apply(lambda x: str(x).replace('*', '').replace('#', '').strip())
        data['input_string'] = [f'Prompt: {data.loc[i, "prompt"]} Title: {data.loc[i, "title"]} Description:{data.loc[i, "description"]}' for i in range(data.shape[0])]
        trained_data = data[data['content'].str.len() <= 5000].reset_index(drop=True)

        # Save the trained data
        self._save_data(trained_data[['input_string', 'content']], encoding)
        return trained_data

# Enum class for different creators
class Creator(Enum):
    LECHILINH = 'LECHILINH'
    PHUNGTANTAI = 'PHUNGTANTAI'
    THAIVANLINH = 'THAIVANLINH'
    CHIHOAI = 'CHIHOAI'

# Class for generating fine-tuned content using the Gemini model
class FineTunedModel:
    __creators = [creator.value for creator in Creator]

    def __init__(self, creator: str | Creator) -> None:
        # Validate the creator type and set the creator
        if isinstance(creator, str):
            if creator.upper() not in self.__creators:
                raise KeyError('Not supported Creator')
            self.creator = creator.upper()
        elif isinstance(creator, Creator):
            self.creator = creator.value.upper()
        else:
            raise TypeError('Invalid type for creator. Must be str or Creator enum.')

        # Initialize the Gemini model
        self.__model = GeminiModel()

    def _get_trained_data(self, data: pd.DataFrame):
        # Extract and format the training data from the DataFrame
        train_data = []
        for i in range(data.shape[0]):
            train_data.append(f'input: {data.loc[i, "input_string"]}'.replace("\\n", "").strip())
            train_data.append(f'output: {data.loc[i, "content"]}'.replace("\\n", "").strip())
        return train_data

    def generate_content(self, request: str = '', video_title: str = '', description: str = ''):
        # Ensure at least one of the input parameters is provided
        if not any([request, video_title, description]):
            raise ValueError('System cannot generate any content without any information')
        print(f"Generating content for {self.creator} with request '{request}', video title '{video_title}', and description '{description}'.")

        # Load the trained data for the specified creator
        data = pd.read_csv(f'TrainedData/{self.creator}/{self.creator}_trained_data.csv')
        
        # Prepare the training data for the model
        texts = self._get_trained_data(data)
        
        # Prepare the input request
        input_request = f'Input: Title: {video_title}, Description: {description}'
        texts.append(input_request)
        texts.append('output:')
        
        # Get the response from the model
        response = self.__model.response(texts, f'Prompt: {request}').removesuffix('```').removeprefix('```').replace('csv', '').strip()
        return response
