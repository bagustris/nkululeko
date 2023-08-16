""" feats_pesq.py
predict PESQ (Perceptual Evaluation of Speech Quality)

adapted from
from https://pytorch.org/audio/main/tutorials/squim_tutorial.html#sphx-glr-tutorials-squim-tutorial-py
paper: https://arxiv.org/pdf/2304.01448.pdf

needs 
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

"""
from nkululeko.util import Util
from nkululeko.featureset import Featureset
import os
import pandas as pd
import os
import nkululeko.glob_conf as glob_conf
import audiofile
import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE

class PESQSet(Featureset):
    """Class to predict PESQ (Perceptual Evaluation of Speech Quality)
    
    """

    def __init__(self, name, data_df):
        """Constructor. is_train is needed to distinguish from test/dev sets, because they use the codebook from the training"""
        super().__init__(name, data_df)
        self.device = self.util.config_val('MODEL', 'device', 'cpu')
        self.model_initialized = False


    def init_model(self):
        # load model
        self.util.debug('loading model...')
        self.objective_model = SQUIM_OBJECTIVE.get_model()
        self.model_initialized = True


    def extract(self):
        """Extract the features or load them from disk if present."""
        store = self.util.get_path('store')
        store_format = self.util.config_val('FEATS', 'store_format', 'pkl')
        storage = f'{store}{self.name}.{store_format}'
        extract = self.util.config_val('FEATS', 'needs_feature_extraction', False)
        no_reuse = eval(self.util.config_val('FEATS', 'no_reuse', 'False'))
        if extract or no_reuse or not os.path.isfile(storage):
            if not self.model_initialized:
                self.init_model()
            self.util.debug('predicting PESQ, this might take a while...')
            emb_series = pd.Series(index = self.data_df.index, dtype=object)
            length = len(self.data_df.index)
            for idx, (file, start, end) in enumerate(self.data_df.index.to_list()):
                signal, sampling_rate = audiofile.read(file, offset=start.total_seconds(), duration=(end-start).total_seconds(), always_2d=True)
                emb = self.get_embeddings(signal, sampling_rate)
                emb_series[idx] = emb
                if idx%10==0:
                    self.util.debug(f'PESQ: {idx} of {length} done')
            self.df = pd.DataFrame(emb_series.values.tolist(), index=self.data_df.index) 
            self.df.columns = ['pesq']
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing predicted PESQ values')
            self.df = self.util.get_store(storage, store_format)
            if self.df.isnull().values.any():
                nanrows = self.df.columns[self.df.isna().any()].tolist()
                print(nanrows)
                self.util.error(f'got nan: {self.df.shape} {self.df.isnull().sum().sum()}')


    def get_embeddings(self, signal, sampling_rate):
        tmp_audio_name = 'pesq_audio_tmp.wav'
        audiofile.write(tmp_audio_name, signal, sampling_rate)
        WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load(tmp_audio_name)
        with torch.no_grad():
            stoi_hyp, pesq_hyp, si_sdr_hyp = self.objective_model(WAVEFORM_SPEECH)
        return float(pesq_hyp[0].numpy())
    
    def extract_sample(self, signal, sr):
        self.init_model()
        feats = self.get_embeddings(signal, sr)
        return feats