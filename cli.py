import argparse
import os
import sys
import hashlib
import json
import yaml
from ml_collections import ConfigDict

from separate import SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR
from gui_data.constants import *

# Constants from UVR.py
if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_PATH, 'models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

model_hash_table = {}

class CliModelData:
    def __init__(self, args):
        self.model_name = args.model
        self.model_path = self.find_model(args.model)
        if not self.model_path:
            raise ValueError(f"Model '{args.model}' not found.")

        self.process_method = self.get_process_method(self.model_path)
        self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]

        # Set defaults from UVR.py or reasonable values
        self.is_gpu_conversion = 0 if args.gpu >= 0 else -1
        self.device_set = str(args.gpu) if args.gpu >= 0 else 'cpu'
        self.is_normalization = True
        self.wav_type_set = 'PCM_16'
        self.mp3_bit_set = '320k'
        self.save_format = 'wav'

        # Stems
        self.primary_stem = args.vocal_stem
        self.secondary_stem = args.instrumental_stem

        # Set default attributes for SeperateAttributes
        self.is_pitch_change = False
        self.semitone_shift = 0.0
        self.is_match_frequency_pitch = False
        self.overlap = 0.25
        self.overlap_mdx = 'Default'
        self.overlap_mdx23 = 8
        self.is_mdx_combine_stems = False
        self.is_mdx_c = False
        self.mdx_c_configs = None
        self.mdxnet_stem_select = 'All Stems'
        self.mixer_path = None
        self.model_samplerate = 44100
        self.model_capacity = (32, 128)
        self.is_vr_51_model = False
        self.is_pre_proc_model = False
        self.is_secondary_model_activated = False
        self.is_secondary_model = False
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_ensemble_mode = False
        self.secondary_model = None
        self.primary_model_primary_stem = self.primary_stem
        self.primary_stem_native = self.primary_stem
        self.is_invert_spec = False
        self.is_deverb_vocals = False
        self.is_mixer_mode = False
        self.secondary_model_scale = None
        self.is_demucs_pre_proc_model_inst_mix = False
        self.ensemble_primary_stem = None
        self.is_multi_stem_ensemble = False
        self.DENOISER_MODEL = None
        self.DEVERBER_MODEL = None
        self.vocal_split_model = None
        self.is_vocal_split_model = False
        self.is_save_inst_vocal_splitter = False
        self.is_inst_only_voc_splitter = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.deverb_vocal_opt = 'Vocals Only'
        self.is_save_vocal_only = False
        self.is_use_opencl = False

        # MDX specific
        self.is_mdx_ckpt = self.model_path.endswith('.ckpt')

        self.get_model_hash()
        if self.process_method == MDX_ARCH_TYPE and self.model_hash:
            self.model_data = self.get_model_data()
            if self.model_data and "config_yaml" in self.model_data:
                self.is_mdx_c = True
                config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                if os.path.isfile(config_path):
                    with open(config_path) as f:
                        self.mdx_c_configs = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        self.is_denoise = False
        self.is_denoise_model = False
        self.is_mdx_c_seg_def = False
        self.mdx_batch_size = 1
        self.compensate = 1.035
        self.mdx_segment_size = 256
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_n_fft_scale_set = None
        self.chunks = 0
        self.margin = 44100

        # Demucs specific
        self.demucs_stems = ALL_STEMS
        self.secondary_model_4_stem = None
        self.secondary_model_4_stem_scale = None
        self.is_chunk_demucs = False
        self.segment = 'Default'
        self.demucs_version = DEMUCS_V4
        self.demucs_source_list = []
        self.demucs_source_map = {}
        self.is_demucs_combine_stems = False
        self.demucs_stem_count = 0
        self.pre_proc_model = None
        self.shifts = 0
        self.is_split_mode = True

        # VR specific
        self.aggression_setting = 0.2
        self.is_tta = False
        self.is_post_process = False
        self.window_size = 512
        self.batch_size = 4
        self.crop_size = 256
        self.is_high_end_process = False
        self.post_process_threshold = 0.2
        self.vr_model_param = '4band_v2'

    def find_model(self, model_name):
        model_paths = [
            os.path.join('models', 'VR_Models', f'{model_name}.pth'),
            os.path.join('models', 'MDX_Net_Models', f'{model_name}.onnx'),
            os.path.join('models', 'MDX_Net_Models', f'{model_name}.ckpt'),
            os.path.join('models', 'Demucs_Models', f'{model_name}.ckpt'),
            os.path.join('models', 'Demucs_Models', 'v3_v4_repo', f'{model_name}.yaml')
        ]
        for path in model_paths:
            if os.path.exists(path):
                return path
        return None

    def get_process_method(self, model_path):
        if 'VR_Models' in model_path:
            return VR_ARCH_PM
        elif 'MDX_Net_Models' in model_path:
            return MDX_ARCH_TYPE
        elif 'Demucs_Models' in model_path:
            return DEMUCS_ARCH_TYPE
        else:
            raise ValueError(f"Could not determine process method for model: {model_path}")

    def get_model_hash(self):
        self.model_hash = None
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash = None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(-10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path, 'rb').read()).hexdigest()
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)

    def get_model_data(self):
        model_settings_json = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            # Fallback for models not in the hash file
            with open(MDX_HASH_JSON, 'r') as json_file:
                hash_mapper = json.load(json_file)
            for hash_string, settings in hash_mapper.items():
                if self.model_hash in hash_string:
                    return settings
        return None

def write_to_console(progress_text, base_text=''):
    print(base_text + progress_text)

def set_progress_bar(step, inference_iterations=0):
    print(f'Progress: {int(step * 100)}%')

def main():
    parser = argparse.ArgumentParser(description='Remove vocals from an audio file.')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input audio file.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('-m', '--model', type=str, required=True, help='Name of the model to use.')
    parser.add_argument('-v', '--vocal_stem', type=str, default='Vocals', help='Name of the vocal stem.')
    parser.add_argument('-i_stem', '--instrumental_stem', type=str, default='Instrumental', help='Name of the instrumental stem.')
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='GPU device to use. -1 for CPU.')

    args = parser.parse_args()

    try:
        model_data = CliModelData(args)
    except ValueError as e:
        print(f"Error: {e}")
        return

    audio_file_base = os.path.splitext(os.path.basename(args.input_path))[0]

    process_data = {
        'export_path': args.output_path,
        'audio_file_base': audio_file_base,
        'audio_file': args.input_path,
        'set_progress_bar': set_progress_bar,
        'write_to_console': write_to_console,
        'process_iteration': lambda: None,
        'cached_source_callback': lambda *args, **kwargs: (None, None),
        'cached_model_source_holder': lambda *args, **kwargs: None,
        'list_all_models': [],
        'is_ensemble_master': False,
        'is_4_stem_ensemble': False,
    }

    separator = None
    if model_data.process_method == VR_ARCH_PM:
        separator = SeperateVR(model_data, process_data)
    elif model_data.process_method == MDX_ARCH_TYPE:
        if model_data.is_mdx_c:
             separator = SeperateMDXC(model_data, process_data)
        else:
             separator = SeperateMDX(model_data, process_data)
    elif model_data.process_method == DEMUCS_ARCH_TYPE:
        separator = SeperateDemucs(model_data, process_data)
    else:
        print(f"Error: Unknown process method '{model_data.process_method}'")
        return

    if separator:
        separator.seperate()

if __name__ == '__main__':
    main()
