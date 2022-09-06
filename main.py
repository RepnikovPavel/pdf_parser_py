from support import run_backend_get_all_unique_words, \
    run_fronted_add_table_with_unique_words_to_html,run_backend_get_tranlate_of_unique_words
import torch

source_pdf_path = "./source.pdf"
output_path = "./source.html"
translate_of_unique_words_path= "./translate_of_unique_words.txt"

sections = run_backend_get_all_unique_words(source_pdf_path, MAX_NUM_OF_OUTPUT=100)
translate = run_backend_get_tranlate_of_unique_words(sections["unusual words"])

torch.save(translate,translate_of_unique_words_path)

translate = torch.load(translate_of_unique_words_path)
run_fronted_add_table_with_unique_words_to_html(translate, output_path)


# all_row_strings = []
# with fitz.open(pdf_path) as pdf:
#     for i, page in enumerate(pdf):
#         words = page.get_text("words")
#         for word in words:
#             all_row_strings.append(word[4])
#         # blocks = page.get_text("blocks")
#         # block_0 = blocks[0]
#         # x_0 = block_0[0]
#         # y_0 =block_0[1]
#         # x_1 =block_0[2]
#         # y_1 =block_0[3]
#         # mat = fitz.Matrix(2, 2)
#         # rect= page.rect
#         # pix = page.get_pixmap(matrix=mat, clip=(x_0,y_0,x_1,y_1))
#         # imaga_path= "./pixmaps/pixmap_{}.png".format(i)
#         # pix.save(imaga_path)