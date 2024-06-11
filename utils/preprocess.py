import pandas as pd
import torch

def data_preprocess(dataset, index):

    question = dataset[index]["question"]
    table_column_names = dataset[index]["table"]["header"]
    table_content_values = dataset[index]["table"]["rows"]

    answer = dataset[index]["answers"]
    answer_list = [str(a).lower() for a in dataset[index]["answers"]]
    answer = f", ".join(answer).lower()

    table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})

    return question, table, answer


def prepare_model_inputs(question, table, highlighted_cells, tokenizer, config):

    tokenized_input = tokenizer(table, question, add_special_tokens = config.tokenizer.add_special_tokens, 
                                padding = config.tokenizer.padding, truncation = config.tokenizer.truncation,  
                                max_length = config.tokenizer.input_max_length, return_tensors = config.tokenizer.return_tensors, 
                                return_token_type_ids = config.tokenizer.return_token_type_ids, 
                                return_attention_mask = config.tokenizer.return_attention_mask)
    
    if isinstance(highlighted_cells, str):
        tokenized_highlighted_cells = []
        hard_relevance_label = torch.zeros((tokenized_input["input_ids"].shape[1]))
        for h_cell in highlighted_cells:
            x = tokenizer(answer = h_cell, add_special_tokens = False,
                                return_tensors = config.tokenizer.return_tensors,
                                return_attention_mask = config.tokenizer.return_attention_mask)["input_ids"].tolist()
            for ele in x[0]:
                hard_relevance_label[tokenized_input["input_ids"].squeeze() == ele] = 1

        highlighted_cells = hard_relevance_label

    index = tokenized_input["input_ids"][0].tolist().index(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" col"))[0])
    highlighted_cells[:index] = 1
    highlighted_cells = torch.tensor(highlighted_cells)

    return tokenized_input, highlighted_cells.unsqueeze(0)


def prepare_parsing_statement_generator_model_inputs(question, tokenizer, table_column_names, table_content_values, config):

    table = "[HEADER] " + " | ".join(table_column_names)
    for row_id, row in enumerate(table_content_values):
        table += f" [ROW] {row_id}: " + " | ".join(row)

    return tokenizer(question, table, add_special_tokens = config.tokenizer.add_special_tokens,
                            padding = config.tokenizer.padding, truncation = config.tokenizer.truncation, 
                            max_length = 1536, return_tensors = config.tokenizer.return_tensors,
                            return_token_type_ids = config.tokenizer.return_token_type_ids,
                            return_attention_mask = config.tokenizer.return_attention_mask)


def prepare_cell_highlighter_model_inputs(parsing_statement, tokenizer, table_column_names, table_content_values, config):

    table = "[HEADER] " + " | ".join(table_column_names)
    for row_id, row in enumerate(table_content_values):
        table += f" [ROW] {row_id}: " + " | ".join(row)
    
    return tokenizer(parsing_statement, table, add_special_tokens = config.tokenizer.add_special_tokens,
                            padding = config.tokenizer.padding, truncation = config.tokenizer.truncation, 
                            max_length = 1536, return_tensors = config.tokenizer.return_tensors,
                            return_token_type_ids = config.tokenizer.return_token_type_ids,
                            return_attention_mask = config.tokenizer.return_attention_mask)