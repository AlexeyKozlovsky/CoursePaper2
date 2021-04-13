import os
import sys
import pytube as pt
import pandas as pd
import numpy as np
sys.path.insert(1, '')
from segmentator import Segmentator
from chunck import Chunck


class Parser:
    def __init__(self, seg, report_path):
        self.seg = seg
        self.report_path = report_path
        self.columns = ['length', 'velocity_x', 'velocity_y']
        
    def parse(self, url, out_path):
        youtube = pt.YouTube(url)
        temp_video_path = 'temp.mp4'
        video = youtube.streams.first().download(filename='temp')
        report = self.seg.get_chuncks(temp_video_path, out_path)
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        return report
        
    def from_cvs(self, csv_path, out_path):
        urls_df = pd.read_csv(csv_path)
        urls_temp = urls_df.to_numpy()
        urls = []
        for url in urls_temp:
            urls.append(url[0])
        urls = np.array(urls, dtype=str)
        report = pd.DataFrame(columns=self.columns)
        
        for url in urls:
            report = pd.concat([report, self.parse(url, out_path)])
            report.to_csv(self.report_path)
        
        return report