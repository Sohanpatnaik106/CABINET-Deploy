import json
import torch
import argparse
import streamlit as st
from easydict import EasyDict
from datasets import load_dataset
from src import HighlightedCluBartModelForGenerativeQuestionAnswering, T5ModelForTableReasoning, T5ModelForTableCellHighlighting
from utils import data_preprocess, prepare_model_inputs, prepare_parsing_statement_generator_model_inputs, prepare_cell_highlighter_model_inputs
from transformers import AutoTokenizer
import pickle
import pandas as pd


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
    # test_dataset = load_dataset(config.data.data_path)["test"]
    initialise_app()
    
    question = st.text_input('Please enter the question')

    table_file = st.file_uploader("upload file", type={"csv", "txt"})
    if table_file is not None:
        table = pd.read_csv(table_file)
        table = table.applymap(str)
        table_column_names = table.columns.tolist()
        table_content_values = table.values.tolist()


    if st.button('Show Content'):
        st.subheader('Question')
        st.write(question)

        st.subheader('Table')
        try:
            st.table(table)
        except ValueError as e:
            st.write("Error displaying table: ", e)
            st.write("Table content:", table)

    # parsing_statement = test_parsing_statements[index]
    # highlighted_cells = test_higlighted_cells[index]
    
    # model_inputs, highlighted_cells = prepare_model_inputs(question = question, highlighted_cells = highlighted_cells, tokenizer = tokenizer, config = config)
    
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


        device = "cuda:5"
        # question, table, answer = data_preprocess(dataset = test_dataset, index = index)
        t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        parsing_statement_generator_model_inputs = prepare_parsing_statement_generator_model_inputs(question, t5_tokenizer, table_column_names, table_content_values, config)

        parsing_statement_generator = T5ModelForTableReasoning(config)
        parsing_statement_generator.load_state_dict(torch.load("./ckpts/cabinet_parsing_statement_generator.pt", map_location = "cpu"))

        parsing_statement_generator.to(device)

        parsing_statement_ids = parsing_statement_generator.model.generate(input_ids = parsing_statement_generator_model_inputs["input_ids"].to(device), 
                                            attention_mask = parsing_statement_generator_model_inputs["attention_mask"].to(device), 
                                        max_new_tokens = 256, num_beams = 3, early_stopping = True).detach().cpu().squeeze()

        parsing_statement = t5_tokenizer.decode(parsing_statement_ids, skip_special_tokens = True)

        st.subheader("Parsing Statement")
        st.write(parsing_statement)

        cell_highlighter_model_inputs = prepare_cell_highlighter_model_inputs(parsing_statement, t5_tokenizer, table_column_names, table_content_values, config)
        cell_highlighter = T5ModelForTableCellHighlighting(config)
        
        cell_highlighter.load_state_dict(torch.load("./ckpts/cabinet_cell_highlighter.pt", map_location = "cpu"))
        cell_highlighter.to(device)

        highlighted_cells_ids = cell_highlighter.model.generate(input_ids = cell_highlighter_model_inputs["input_ids"].to(device), 
                                                    attention_mask = cell_highlighter_model_inputs["attention_mask"].to(device),
                                      max_new_tokens = 256, num_beams = 3, early_stopping = True).squeeze().detach().cpu()

        highlighted_cells = t5_tokenizer.decode(highlighted_cells_ids, skip_special_tokens=True)
        st.subheader("Predicted Cells")
        st.write(highlighted_cells)

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)

        model_inputs, highlighted_cells = prepare_model_inputs(question = question, table = table, highlighted_cells = highlighted_cells, tokenizer = tokenizer, config = config)
        model = HighlightedCluBartModelForGenerativeQuestionAnswering(config)
        model.load_state_dict(torch.load("./ckpts/cabinet_wikitq_ckpt.pt", map_location = "cpu"))
        model.to(device)

        input_ids, attention_mask, token_type_ids, highlighted_cells = \
            model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["token_type_ids"], highlighted_cells
        output_ids = model.model.generate(input_ids = input_ids.to(device), max_new_tokens = config.tokenizer.output_max_length, 
                                        num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(device), 
                                        highlighted_cells = highlighted_cells.to(device))
        
        predicted_answer = tokenizer.decode(output_ids.squeeze(), skip_special_tokens = True)
        st.subheader("Generated Answer")
        st.write(predicted_answer)



        # device = "cuda:3"
        # model = HighlightedCluBartModelForGenerativeQuestionAnswering(config)
        # model.load_state_dict(torch.load("./ckpts/cabinet_wikitq_ckpt.pt", map_location = "cpu"))
        # model.to(device)

        # input_ids, attention_mask, token_type_ids, highlighted_cells = \
        #     model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["token_type_ids"], highlighted_cells
        # output_ids = model.model.generate(input_ids = input_ids.to(device), max_new_tokens = config.tokenizer.output_max_length, 
        #                                 num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(device), 
        #                                 highlighted_cells = highlighted_cells.to(device))
        
        # predicted_answer = tokenizer.decode(output_ids.squeeze(), skip_special_tokens = True)
        # st.subheader("Parsing Statement")
        # st.write(parsing_statement)

        # # st.subheader("Highlighted Cells")
        # # st.write(highlighted_cells_text)

        # st.subheader("Generated Answer")
        # st.write(predicted_answer)

        # st.subheader("Actual Answer")
        # st.write(answer)