import json
import torch
import argparse
import streamlit as st
from easydict import EasyDict
from datasets import load_dataset
from src import HighlightedCluBartModelForGenerativeQuestionAnswering
from utils import data_preprocess, prepare_model_inputs
from transformers import AutoTokenizer
import pickle

def process_config(config: dict, args = None):

    if args is not None:
        args_dict = vars(args)
        merged_dict = {**args_dict, **config}
    else:
        merged_dict = config

    merged_config = EasyDict(merged_dict)
    return merged_config


def initialise_app():
    st.title("Table Questions Answering")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "config.json", type = str, help = "Path to experiment configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config = process_config(config, args)
    test_dataset = load_dataset(config.data.data_path)["test"]

    with open("./dataset/test_wiki_tq_reason_without_answer_flant5.pkl", "rb") as f:
        test_parsing_statements = pickle.load(f)

    with open("./dataset/wiki_tq_test_highlighted_cell.pkl", "rb") as f:
        test_higlighted_cells = pickle.load(f)

    initialise_app()
    index = st.number_input('Enter the index:', min_value = 0, max_value = len(test_dataset["question"]) - 1, step = 1)
    question, table, answer = data_preprocess(dataset = test_dataset, index = index)
    parsing_statement = test_parsing_statements[index]
    highlighted_cells = test_higlighted_cells[index]

    if st.button('Show Content'):
        if index >= 0 and index < len(test_dataset["question"]):
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

    
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    model_inputs, highlighted_cells = prepare_model_inputs(question = question, table = table, highlighted_cells = highlighted_cells, tokenizer = tokenizer, config = config)
    
    # print(highlighted_cells.shape)
    if st.button('Generate Answer'):

        st.subheader('Question')
        st.write(question)

        st.subheader('Table')
        try:
            st.table(table)
        except ValueError as e:
            st.write("Error displaying table: ", e)
            st.write("Table content:", table)

        device = "cuda:3"
        model = HighlightedCluBartModelForGenerativeQuestionAnswering(config)
        model.load_state_dict(torch.load("./ckpts/cabinet_wikitq_ckpt.pt", map_location = "cpu"))
        model.to(device)

        input_ids, attention_mask, token_type_ids, highlighted_cells = \
            model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["token_type_ids"], highlighted_cells
        output_ids = model.model.generate(input_ids = input_ids.to(device), max_new_tokens = config.tokenizer.output_max_length, 
                                        num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(device), 
                                        highlighted_cells = highlighted_cells.to(device))
        
        predicted_answer = tokenizer.decode(output_ids.squeeze(), skip_special_tokens = True)
        st.subheader("Parsing Statement")
        st.write(parsing_statement)

        # st.subheader("Highlighted Cells")
        # st.write(highlighted_cells_text)

        st.subheader("Generated Answer")
        st.write(predicted_answer)

        st.subheader("Actual Answer")
        st.write(answer)