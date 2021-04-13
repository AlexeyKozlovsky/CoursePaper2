import sys
sys.path.insert(1, '/')
from chunck import Chunck
from segmentator import Segmentator


if __name__ == '__main__':
    seg = Segmentator('../Models/vosk-model-small-ru-0.4', 
                      '../Models/shape_predictor_68_face_landmarks.dat')
                      

    seg.get_chuncks('temp.mp4', 'DataSet/')
