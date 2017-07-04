from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

def char2id(char, first_letter): #'a'->1, ...,'z'->25 and ' '->0 #first_letter = ord('a')
	if char in string.ascii_lowercase:
		return ord(char) - first_letter + 1
	elif char == ' ':
		return 0
	else:
		print('Unexpected character: %s' %char)
		return 0

def id2char(dictid, first_letter): # 1 -> 'a',..., 25 -> 'z' and other number ->' '
	if dictid > 0:
		# char is the inverse of ord
		return chr(dictid + first_letter - 1)
	else:
		return ' '

