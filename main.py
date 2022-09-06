import time

from support import run_backend_get_all_unique_words, \
    run_fronted_add_table_with_unique_words_to_html, run_backend_get_tranlate_of_unique_words, run_backend_get_metada, \
    run_frontend_start_html_code, run_frontend_end_html_code, run_fronted_add_metadat_of_text_to_html
import torch

source_pdf_path = "./source.pdf"
output_path = "./table_of_words.html"
translate_of_unique_words_path= "./translate_of_unique_words.txt"
start_time = time.time()
num_of_words_in_table =100
print("time for execution {} m".format(int(num_of_words_in_table*10.0/60.0)))
sections = run_backend_get_all_unique_words(source_pdf_path, MAX_NUM_OF_OUTPUT=num_of_words_in_table)
translate = run_backend_get_tranlate_of_unique_words(sections["unusual words"])
torch.save(translate,translate_of_unique_words_path)

translate = torch.load(translate_of_unique_words_path)

metadata_of_pdf = run_backend_get_metada(source_pdf_path)

html_code = run_frontend_start_html_code()

html_code = run_fronted_add_metadat_of_text_to_html(html_code, metadata_of_pdf)
html_code = run_fronted_add_table_with_unique_words_to_html(html_code, translate)

run_frontend_end_html_code(html_code, output_path)

stop_time = time.time()
print("time of execution of program is {} s".format(stop_time-start_time))

