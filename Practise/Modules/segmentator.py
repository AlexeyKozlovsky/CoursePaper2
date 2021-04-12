import os
import subprocess
import json
import math

import cv2
import vosk
import librosa
import pytube as pt
import numpy as np
import pandas as pd

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

class Segmentator:
	"""Класс для сегментации видоефайлов"""
	def __init__(self, conf):
		self.conf = conf # Вероятность, начиная с которой надо оставлять распознанные слов
		self.word_dict = None
		

	def segment_to_chuncks(self, path, out_path):
		"""
		Метод для сегментации видеофайла по словам в датасет

		path: путь для видео
		out_path: путь для датасета
		"""
		

		pass

	def _extract_words(self, res):
		jres = json.loads(res)
		if not 'result' in jres:
			return []
		words = jres['result']
		return words

	def _transcribe_words(self, recognizer, bytes):
		result = []

		chunck_size = 4000
		for chunck_no in range(math.ceil(len(bytes)) / chunck_size):
			start = chunck_size * chunck_no
			end = min(len(bytes), (chunck_no + 1) * chunck_size)
			data = bytes[start : end]

			if recognizer.AcceptWaveform(data):
				words = self._extract_words(recognizer.Result())
				result += words

		result += self._extract_words(recognizer.FinalResult())

		return result

	def to_cvs(self, model_path, input_path, out_path=None):
		"""
		Метод для сбора информации о таймингах слов в
		словарь или cvs файл

		model_path: путь до модели для распознавания речи
		input_path: путь к видеофайлу
		out_path: путь к csv файлу. Если None, то сохранять не надо

		returns: словарь слов с таймингами и вероятностями
		"""

		vosk.SetLogLevel(-1)

		sample_rate = 16000
		audio, sr = librosa.load(input_path, sr=sample_rate)

		int16 = np.int16(audio * 32768).tobytes()

		model = vosk.Model(model_path)
		recognizer = vosk.KaldiRecognizer(model, sample_rate)

		res = transcribe_words(recognizer, int16)
		df = pd.DataFrame.from_records(res)
		df = df.sort_values('start')

		if out_path is not None:
			df.to_cvs(out_path, index=False)


		word_dict = {}
		for index, row in df.iterrows():
			word_dict[row['word']] = []

		for index, row in df.iterrows():
			word_dict[row['word']].append((row['start'], row['end'], row['conf']))

		return word_dict







			

