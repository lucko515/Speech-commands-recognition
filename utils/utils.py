import numpy as np

def character_mapper(char_map = " abcdefghijklmnopqrstuvwxyz'"):
	'''
	Creates two dictionaries that are used to convert words to Integer arrays and vice versa.
	

	:params: 
		char_map - String, all characters that may occure in resulting words
	
	:return:
		id_to_char - Python dictionary, maps integer array to string representation (in respect to the char_map)
		char_to_id - Python dictionary, maps characters to integer representation (in respect to the char_map)
	'''

    id_to_char = {i+1:j for i,j in enumerate(char_map)}
    char_to_id = {j:i+1 for i,j in enumerate(char_map)}
    
    return id_to_char, char_to_id


def word_to_int(word, char_to_id):
	'''
	Converts word to its Integer representation

	:params:
		word - String, input word that is converted to an integer representation
		char_to_id - Python dictionary, maps characters to integer representation (in respect to the char_map)

	:returns:
		numpy array with an integer representation of the input word
	'''
    return np.array([char_to_id[c] for c in word])


def int_to_word(word_array, id_to_char):
	'''
	Converts an Integer array to its word/string representation.

	:params:
		word_array - Numpy array, an Integer representation that is used to 'recover' a word
		id_to_char - Python dictionary, maps integer array to string representation (in respect to the char_map)

	:returns:
		result_string - String, resulting word/string representation of the input word_array
	'''

	result_string = ""
	for c in word:
		result_string += id_to_char[c] 
    return result_string