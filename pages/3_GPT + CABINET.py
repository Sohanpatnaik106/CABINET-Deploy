import re
import json
import torch
import pickle
import random
import requests
import argparse
import unicodedata
import numpy as np
import pandas as pd
import random as rn
from tqdm import tqdm
import streamlit as st
from codecs import open
from easydict import EasyDict
from math import isnan, isinf
from datasets import load_dataset
from transformers import AutoTokenizer
from abc import ABCMeta, abstractmethod
from utils import data_preprocess, prepare_model_inputs
from src import HighlightedCluBartModelForGenerativeQuestionAnswering

# GPT Credentials
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
AUTH_KEY = st.secrets["AUTH_KEY"]
IMS_URL = st.secrets["IMS_URL"]
AZURE_CHAT_COMPLETION_START = st.secrets["AZURE_CHAT_COMPLETION_START"] 
AZURE_CHAT_COMPLETION = st.secrets["AZURE_CHAT_COMPLETION"] 
AZURE_COMPLETIONS = st.secrets["AZURE_COMPLETIONS"]

# Set global seed
def set_seed(seed):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def process_config(config: dict, args = None):

    if args is not None:
        args_dict = vars(args)
        merged_dict = {**args_dict, **config}
    else:
        merged_dict = config

    merged_config = EasyDict(merged_dict)
    return merged_config


def normalize(x):

    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')

    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)

    while True:
        
        old_x = x

        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        
        if x == old_x:
            break
    
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    
    return x


class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None

class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):

        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        
        self._year = year
        self._month = month
        self._day = day
        
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))

    __repr__ = __str__

    def match(self, other):
        
        assert isinstance(other, Value)
        
        if self.normalized == other.normalized:
            return True
        
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """

    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    
    if not corenlp_value:
        corenlp_value = original_string
    
    # Number?
    amount = NumberValue.parse(corenlp_value)
    
    if amount is not None:
        return NumberValue(amount, original_string)
    
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                        in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """

    # Check size
    if len(target_values) != len(predicted_values):
        return False
    
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    
    return True


def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')


def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)

    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split('|')]


def get_response(prompt):
    params = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': AUTH_KEY,
        'grant_type': 'authorization_code',
    }
    response = requests.post(IMS_URL, data=params)
    temp_auth_token=json.loads(response.text)['access_token']

    def get_openai_response(azure_url, json_data, temp_auth_token):
        headers = {
            'x-gw-ims-org-id': CLIENT_ID,
            'x-api-key': CLIENT_ID,
            'Authorization': f'Bearer {temp_auth_token}',
            'Content-Type': 'application/json',
        }
        response = requests.post(azure_url, headers=headers, json=json_data)
        return json.loads(response.text)

    # json_data = {
    #     "messages": [{
    #         "role": "system",
    #         "content": "You are an AI assistant that helps people answer their queries."
    #     }],
    #     "llm_metadata": {
    #         "model_name": "gpt-35-turbo-1106",
    #         "temperature": 0.2,
    #         "max_tokens": 256,
    #         "top_p": 1.0,
    #         "frequency_penalty": 0,
    #         "presence_penalty": 0,
    #         "llm_type": "azure_chat_openai"
    #     },
    # }
    # openai_response = get_openai_response(AZURE_CHAT_COMPLETION_START, json_data, temp_auth_token)
    # conversation_id = openai_response['conversation_id']

    json_data={
        "messages":[
            {
                "role": "user",
                "content": prompt
            }
        ],
        "llm_metadata": {
            "model_name": "gpt-35-turbo-1106",
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "llm_type": "azure_chat_openai",
            "n": 1
        },
    }
    try:
        openai_response = get_openai_response(AZURE_CHAT_COMPLETION, json_data, temp_auth_token)
    except:
        openai_response = {}

    if "generations" not in list(openai_response.keys()):
        return ""
    return openai_response["generations"][0][0]["text"]

def process_table(table_column_names, table_content_values):
    table = "[HEADER] " + " | ".join(table_column_names)
    for row_id, row in enumerate(table_content_values):
        table += f" [ROW] {row_id}: " + " | ".join(row)

    return table

def get_relevance_scores(question, table, tokenizer, model, config, device, highlighted_cells):

    tokenized_input = tokenizer(table, question, add_special_tokens = config.tokenizer.add_special_tokens, 
                                padding = config.tokenizer.padding, truncation = config.tokenizer.truncation,  
                                max_length = config.tokenizer.input_max_length, return_tensors = config.tokenizer.return_tensors, 
                                return_token_type_ids = config.tokenizer.return_token_type_ids, 
                                return_attention_mask = config.tokenizer.return_attention_mask)

    input_ids = tokenized_input["input_ids"].squeeze()

    index = tokenized_input["input_ids"][0].tolist().index(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" col"))[0])
    highlighted_cells[:index] = 1
    highlighted_cells = torch.tensor(highlighted_cells).unsqueeze(0).to(device)

    decomposer_outputs = model.model.model.decomposer(input_ids=tokenized_input["input_ids"].to(device),
                                                attention_mask=tokenized_input["attention_mask"].to(device),
                                                head_mask=None,
                                                inputs_embeds=None,
                                                output_attentions=None,
                                                output_hidden_states=None,
                                                return_dict=None,
        )

    latent_rep = model.model.model.latent_rep_head(decomposer_outputs[0])
    soft_labels_numerator = (1 + torch.norm((latent_rep.unsqueeze(2) - model.model.model.cluster_centers.unsqueeze(0).unsqueeze(0)), dim = -1) / model.model.model.clu_alpha) ** (-(1 + model.model.model.clu_alpha) / 2)
    soft_labels = soft_labels_numerator / torch.sum(soft_labels_numerator, dim = -1).unsqueeze(-1)

    token_scores_1 = model.model.model.token_classifier_score1(latent_rep)
    token_scores_2 = model.model.model.token_classifier_score2(latent_rep)
    gaussian_rvs = model.model.gaussian_dist.sample(token_scores_1.shape).to(token_scores_1.device)
    relevance_logit = gaussian_rvs * token_scores_1 + token_scores_2
    relevance_score = model.model.model.sigmoid(relevance_logit)
    gaussian_relevance_score = model.model.stricter_gaussian_dist.log_prob(relevance_logit).exp().to(token_scores_1.device)

    relevance_score = 0.7 * relevance_score + 0.3 * highlighted_cells.unsqueeze(-1)

    return input_ids, relevance_score.squeeze()


def create_top_k_mask(arr, k):
    flat = arr.flatten()
    top_k_indices = np.argpartition(flat, -k)[-k:]

    mask_flat = np.zeros(flat.shape, dtype=bool)
    mask_flat[top_k_indices] = True

    mask = mask_flat.reshape(arr.shape)
    return mask

def construct_relevant_cells_table(input_ids, table_column_names, table_content_values, relevance_score, tokenizer):

    # for x in input_ids:
    #     print(x, tokenizer.decode(x))

    # exit()

    row_index = -1
    col_index = -1
    cell_relevance_scores = np.zeros((len(table_content_values), len(table_column_names)))

    for i, index in enumerate(input_ids):

        if row_index >= len(cell_relevance_scores):
            break

        if index == 2:
            break

        if index == 3236:
            row_index += 1
            col_index = 0

        elif index == 4832 or index == 1721:
            if index == 1721:
                col_index += 1
        
            cell_score = 0
            cell_token_count = 0
            for j in range(i+1, len(input_ids)):
                if input_ids[j] == 3236 or input_ids[j] == 1721 or input_ids[j] == 2:
                    if row_index >= len(cell_relevance_scores) or col_index >= len(cell_relevance_scores[0]):
                        cell_score = 0
                        cell_token_count = 0
                        break
                    else:
                        cell_relevance_scores[row_index][col_index] = cell_score / (cell_token_count + 1)
                        cell_score = 0
                        cell_token_count = 0
                        break
                else:
                    cell_score += relevance_score[j]
                    cell_token_count += 1
            
        else:
            continue

    # print(cell_relevance_scores)
    # cell_importance = create_top_k_mask(cell_relevance_scores, 20)
    cell_importance = create_top_k_mask(cell_relevance_scores, int(0.3 * len(table_column_names) * len(table_content_values)))

    table = "[HEADER] " + " | ".join(table_column_names)
    for row_id, row in enumerate(table_content_values):
        table += f" [ROW] {row_id}: "
        for col_id, cell in enumerate(row):
            if cell_importance[row_id][col_id]:
                table +=  cell + " (important cell) | "
            else:
                table += cell + " | "

    table_df = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
    for i in range(len(cell_importance)):
        for j in range(len(cell_importance[0])):
            table_df.iloc[i, j] = table_df.iloc[i, j] + " (important cell)"

    return table, table_df



def prepare_few_shot_examples_gpt_cabinet(train_questions, train_answers, train_tables, indices, highlighted_cells_list, model, tokenizer, 
                              config, device, train_table_column_names, train_table_content_values):

    assert len(indices) == 3, "Please provide exactly three indices to be used as in-context examples"
    
    table_content_values = train_table_content_values[indices[0]]
    table_column_names = train_table_column_names[indices[0]]
    example1_table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
    example1_highlighted_cells = highlighted_cells_list[0]
    example1_input_ids, example1_relevance_score = get_relevance_scores(question, example1_table, tokenizer, model, config, device, example1_highlighted_cells)
    example1_relevant_table, _ = construct_relevant_cells_table(example1_input_ids, table_column_names, table_content_values, example1_relevance_score, tokenizer)
    example1 = f"Question: {train_questions[indices[0]].lower()}\n\nTable: {example1_relevant_table.lower()}\n\nAnswer: {train_answers[indices[0]].lower()}"

    table_content_values = train_table_content_values[indices[1]]
    table_column_names = train_table_column_names[indices[1]]
    example2_table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
    example2_highlighted_cells = highlighted_cells_list[1]
    example2_input_ids, example2_relevance_score = get_relevance_scores(question, example2_table, tokenizer, model, config, device, example2_highlighted_cells)
    example2_relevant_table, _ = construct_relevant_cells_table(example2_input_ids, table_column_names, table_content_values, example2_relevance_score, tokenizer)
    example2 = f"Question: {train_questions[indices[1]].lower()}\n\nTable: {example2_relevant_table.lower()}\n\nAnswer: {train_answers[indices[1]].lower()}"

    table_content_values = train_table_content_values[indices[2]]
    table_column_names = train_table_column_names[indices[2]]
    example3_table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
    example3_highlighted_cells = highlighted_cells_list[2]
    example3_input_ids, example3_relevance_score = get_relevance_scores(question, example3_table, tokenizer, model, config, device, example3_highlighted_cells)
    example3_relevant_table, _ = construct_relevant_cells_table(example3_input_ids, table_column_names, table_content_values, example3_relevance_score, tokenizer)
    example3 = f"Question: {train_questions[indices[2]].lower()}\n\nTable: {example3_relevant_table.lower()}\n\nAnswer: {train_answers[indices[2]].lower()}"

    return example1, example2, example3


def prepare_test_table(question, tokenizer, model, config, device, highlighted_cells, table_content_values, table_column_names):

    table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
    input_ids, relevance_score = get_relevance_scores(question, table, tokenizer, model, config, device, highlighted_cells)
    relevant_table, relevant_table_df = construct_relevant_cells_table(input_ids, table_column_names, table_content_values, relevance_score, tokenizer)

    return relevant_table, relevant_table_df


def prepare_few_shot_examples_gpt_only(train_questions, train_answers, train_tables, indices):

    assert len(indices) == 3, "Please provide exactly three indices to be used as in-context examples"


    
    example1 = f"Question: {train_questions[indices[0]].lower()}\n\nTable: {train_tables[indices[0]].lower()}\n\nAnswer: {train_answers[indices[0]].lower()}"
    example2 = f"Question: {train_questions[indices[1]].lower()}\n\nTable: {train_tables[indices[1]].lower()}\n\nAnswer: {train_answers[indices[1]].lower()}"
    example3 = f"Question: {train_questions[indices[2]].lower()}\n\nTable: {train_tables[indices[2]].lower()}\n\nAnswer: {train_answers[indices[2]].lower()}"

    return example1, example2, example3


def initialise_app():
    st.title("Table Questions Answering")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "config.json", type = str, help = "Path to experiment configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config = process_config(config, args)
    if "dataset" not in st.session_state.keys():
        st.session_state["dataset"] = load_dataset("wikitablequestions")
    
    dataset = st.session_state["dataset"]

    if "train_questions" not in st.session_state.keys():
        st.session_state["train_questions"] = dataset["train"]["question"]
    train_questions = st.session_state["train_questions"]

    if "train_answers" not in st.session_state.keys():
        st.session_state["train_answers"] = dataset["train"]["answers"]
        st.session_state["train_answers"] = [" ".join(ans).lower() for ans in st.session_state["train_answers"]]
    train_answers = st.session_state["train_answers"]

    if "train_table_column_names" not in st.session_state.keys():
        st.session_state["train_table_column_names"] = [dataset["train"][i]["table"]["header"] for i in range(len(train_questions))]
    train_table_column_names = st.session_state["train_table_column_names"]
    
    if "train_table_content_values" not in st.session_state.keys():
        st.session_state["train_table_content_values"] = [dataset["train"][i]["table"]["rows"] for i in range(len(train_questions))]
    train_table_content_values = st.session_state["train_table_content_values"]

    if "train_tables" not in st.session_state.keys():
        st.session_state["train_tables"] = [process_table(train_table_column_names[i], train_table_content_values[i]) for i in range(len(train_table_column_names))]

    train_tables = st.session_state["train_tables"]

    if "train_highlighted_cells" not in st.session_state.keys():
        with open("/disks/1/milan/tabllm_demo/code/dataset/wiki_tq_train_highlighted_cell.pkl", "rb") as f: 
            st.session_state["train_highlighted_cells"] = pickle.load(f)
    train_highlighted_cells = st.session_state["train_highlighted_cells"]

    if "test_questions" not in st.session_state.keys():
        st.session_state["test_questions"] = dataset["test"]["question"]
    test_questions = st.session_state["test_questions"]
    
    if "test_answers" not in st.session_state.keys():
        test_answers = dataset["test"]["answers"]
        st.session_state["test_answers"] = [" ".join(ans).lower() for ans in test_answers]
    test_answers = st.session_state["test_answers"]


    if "test_table_column_names" not in st.session_state.keys():
        st.session_state["test_table_column_names"] = [dataset["test"][i]["table"]["header"] for i in range(len(test_questions))]
    test_table_column_names = st.session_state["test_table_column_names"]
    
    if "test_table_content_values" not in st.session_state.keys():
        st.session_state["test_table_content_values"] = [dataset["test"][i]["table"]["rows"] for i in range(len(test_questions))]
    test_table_content_values = st.session_state["test_table_content_values"]

    if "test_tables" not in st.session_state.keys():
        st.session_state["test_tables"] = [process_table(test_table_column_names[i], test_table_content_values[i]) for i in range(len(test_table_column_names))]

    test_tables = st.session_state["test_tables"]

    if "test_highlighted_cells" not in st.session_state.keys():
        with open("/disks/1/milan/tabllm_demo/code/dataset/wiki_tq_test_highlighted_cell.pkl", "rb") as f: 
            st.session_state["test_highlighted_cells"] = pickle.load(f)
    test_highlighted_cells = st.session_state["test_highlighted_cells"]
    
    if "test_parsing_statements" not in st.session_state.keys():
        with open("./dataset/test_wiki_tq_reason_without_answer_flant5.pkl", "rb") as f:
            st.session_state["test_parsing_statements"] = pickle.load(f)
    test_parsing_statements = st.session_state["test_parsing_statements"]

    if "model" not in st.session_state.keys():
        device = "cuda:3"
        model = HighlightedCluBartModelForGenerativeQuestionAnswering(config)
        model.load_state_dict(torch.load("../code/ckpts/cabinet_wikitq_ckpt.pt", map_location = "cpu"))
        model.to(device)
        st.session_state["device"] = device
        st.session_state["model"] = model
    model = st.session_state["model"]
    device = st.session_state["device"]
    
    if "tokenizer" not in st.session_state.keys():
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained("neulab/omnitab-large")
    tokenizer = st.session_state["tokenizer"]

    initialise_app()
    index = st.number_input('Enter the index:', min_value = 0, max_value = len(dataset["test"]["question"]) - 1, step = 1)
    question, table, answer = data_preprocess(dataset = dataset["test"], index = index)
    parsing_statement = test_parsing_statements[index]
    highlighted_cells = test_highlighted_cells[index]

    if st.button('Show Content'):
        if index >= 0 and index < len(dataset["test"]["question"]):
            st.subheader('Question')
            st.write(question)

            st.subheader('Table')
            try:
                st.table(table)
            except ValueError as e:
                st.write("Error displaying table: ", e)
                st.write("Table content:", table)
                  
        else:
            st.write('Index out of range')

    seed_gpt_only_prompt = "You will given 3 examples of how to derive the answers to questions corresponding to tables. In a similar fashion as shown in the examples, provide the correct answer to the question for the given table.\n\nEXAMPLE 1\n\n--------------\n\n{example1}\n\nEXAMPLE 2\n\n--------------\n\n{example2}\n\nEXAMPLE 3\n\n--------------\n\n{example3}\n\nKeeping in mind the above examples, now provide the answer to the given question for the corresponding table.\n\nQuestion: {question}\n\nTable: {table}\n\nAnswer: "
    seed_gpt_cabinet_prompt = "You will given 3 examples of how to derive the answers to questions corresponding to tables. Certain cells in the table will be marked important, which implies that those cells have higher relevance for answering the question. In a similar fashion as shown in the examples, provide the two-three word correct answers to the question for the given table.\n\nEXAMPLE 1\n\n--------------\n\n{example1}\n\nEXAMPLE 2\n\n--------------\n\n{example2}\n\nEXAMPLE 3\n\n--------------\n\n{example3}\n\nKeeping in mind the above examples, now provide the answer to the given question for the corresponding table.\n\nQuestion: {question}\n\nTable: {table}\n\nAnswer: "
    
    # print(highlighted_cells.shape)
    if st.button('Generate Answer Only with GPT-3.5-Turbo'):

        st.subheader('Question')
        st.write(question)

        st.subheader('Table')
        try:
            st.table(table)
        except ValueError as e:
            st.write("Error displaying table: ", e)
            st.write("Table content:", table)

        
        indices = [random.randint(0, len(train_questions) // 3), random.randint(len(train_questions) // 3, 2 * len(train_questions) // 3), random.randint(2 * len(train_questions) // 3, len(train_questions) - 1)]

        example1, example2, example3 = prepare_few_shot_examples_gpt_only(train_questions, train_answers, train_tables, indices)
        prompt = seed_gpt_only_prompt.format(example1 = example1, example2 = example2, example3 = example3, question = question, table = test_tables[index])

        response = get_response(prompt)
        gpt_only_response = response.lower()

        st.subheader("GPT-Only Generated Answer")
        st.write(gpt_only_response)

        st.subheader("Actual Answer")
        st.write(answer)

    if st.button('Generate Answer with GPT-3.5-Turbo + CABINET'):

        st.subheader('Question')
        st.write(question)

        st.subheader('Table')
        try:
            st.table(table)
        except ValueError as e:
            st.write("Error displaying table: ", e)
            st.write("Table content:", table)
            
        indices = [random.randint(0, len(train_questions) // 3), random.randint(len(train_questions) // 3, 2 * len(train_questions) // 3), random.randint(2 * len(train_questions) // 3, len(train_questions) - 1)]
        highlighted_cells_list = [train_highlighted_cells[indices[0]], train_highlighted_cells[indices[1]], train_highlighted_cells[indices[2]]]

        example1, example2, example3 = prepare_few_shot_examples_gpt_cabinet(train_questions, train_answers, train_tables, indices, highlighted_cells_list, 
                                                                model, tokenizer, config, device, train_table_column_names, train_table_content_values)

        test_table, test_table_df = prepare_test_table(question, tokenizer, model, config, device, test_highlighted_cells[index], test_table_content_values[index], test_table_column_names[index])
        prompt = seed_gpt_cabinet_prompt.format(example1 = example1, example2 = example2, example3 = example3, question = question, table = test_table)

        response = get_response(prompt)
        gpt_with_cabinet_response = response.lower()

        st.subheader("GPT-Only Generated Answer")
        st.write(gpt_only_response)

        st.subheader('Table with Important Cells Marked')
        try:
            st.table(test_table_df)
        except ValueError as e:
            st.write("Error displaying table: ", e)
            st.write("Table content:", test_table_df)

        st.subheader("GPT With CABINET Generated Answer")
        st.write(gpt_with_cabinet_response)

        st.subheader("Actual Answer")
        st.write(answer)
        