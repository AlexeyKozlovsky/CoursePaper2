import os
import sys
import subprocess
import json
import math

import cv2
import vosk
import librosa
import pytube as pt
import numpy as np
import pandas as pd
import moviepy.editor as mp

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

sys.path.insert(1, '../Modules/')
from chunck import Chunck


class Segmentator:
	"""Класс для сегментации видоефайлов"""
	def __init__(self, audio_model_path, landmarks_model_path, conf=1):
		self.landmarks_model_path = landmarks_model_path
		self.audio_model_path = audio_model_path
		self.conf = conf # Вероятность, начиная с которой надо оставлять распознанные слов
		self.word_dict = None
		

	def segment_to_chuncks(self, path, out_path):
		"""
		Метод для сегментации видеофайла по словам в датасет

		path: путь для видео
		out_path: путь для датасета
		"""
		if self.word_dict is None:
			self.word_dict = self.segment_words(path)

		for key in self.word_dict:
			for index, word in enumerate(self.word_dict[key]):
				max_index = 0

				if word[2] >= self.conf:
					l_path = f'{out_path}/{word[0]}'
					w_path = f'{out_path}/{word[0]}/{word}'

					if not os.path.exist(l_path):
						os.mkdir(l_path)
					if not os.path.exist(w_path):
						os.mkdir(w_path)
					else:

						files = os.listdir()
						for file in files:
							max_index = max(max_index, file.partition('.'))

					temp_path = f'{w_path}/{max_index}t.mp4'
					ffmpeg_extract_subclip(path, word[0], word[1], temp_path)

					ch = Chunck(path, self.landmarks_model_path)
					ch.prepare()
					ch.to_file(f'{w_path}/{max_index}.mp4')

					if os.path.isfile(temp_path):
						os.remove(temp_path)
						

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

	def segment_words(self, input_path, out_path=None):
		"""
		Метод для сбора информации о таймингах слов в
		словарь или cvs файл

		input_path: путь к видеофайлу
		out_path: путь к csv файлу. Если None, то сохранять не надо

		returns: словарь слов с таймингами и вероятностями
		"""

		clip = mp.VideoFileClip(input_path)
		temp_audio_path = 'audio.mp3'
		clip.audio.write_audiofile(temp_audio_path)

		vosk.SetLogLevel(-1)

		sample_rate = 16000
		audio, sr = librosa.load(temp_audio_path, sr=sample_rate)

		int16 = np.int16(audio * 32768).tobytes()

		model = vosk.Model(self.audio_model_path)
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

		if os.path.isfile(temp_audio_path):
			os.delete(temp_audio_path)

		return word_dict
