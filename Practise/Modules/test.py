import pytube as pt
import sys
sys.path.insert(1, '')
from chunck import Chunck
from segmentator import Segmentator
from parser import Parser

if __name__ == '__main__':
    seg = Segmentator('../Models/shape_predictor_68_face_landmarks.dat',
                      '../Models/vosk-model-small-ru-0.4')
    p = Parser(seg, 'report.csv')
    p.from_cvs('test.csv', 'report.csv')

