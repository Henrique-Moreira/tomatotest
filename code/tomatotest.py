import io
from PIL import Image
from datasets import GeneratorBasedBuilder, DatasetInfo, Features, SplitGenerator, Value, Array2D, Split
import datasets
import numpy as np
import h5py
from huggingface_hub import HfFileSystem
import os

class CustomConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(CustomConfig, self).__init__(**kwargs)
        self.dataset_type = kwargs.pop("name", "all")
        self.local_data_dir = kwargs.pop("local_data_dir", None) # Adicione esta linha
        self.local_train_txt = kwargs.pop("local_train_txt", None) # Adicione esta linha
        self.local_val_txt = kwargs.pop("local_val_txt", None) # Adicione esta linha
_metadata_urls = {
    "train":"https://huggingface.co/datasets/XingjianLi/tomatotest/resolve/main/train.txt",
    "val":"https://huggingface.co/datasets/XingjianLi/tomatotest/resolve/main/val.txt"
}

# ADICIONE ESTE BLOCO NO SEU tomatotest.py, LOGO ABAIXO DE _metadata_urls
_archives_urls = {
    "full": [f"https://huggingface.co/datasets/XingjianLi/tomatotest/resolve/main/data/archive_{i}.tar" for i in range(1, 161)],
    "sample": ["https://huggingface.co/datasets/XingjianLi/tomatotest/resolve/main/data/archive_0.tar"],
    "depth": ["https://huggingface.co/datasets/XingjianLi/tomatotest/resolve/main/data/archive_0.tar"],
    "seg": ["https://huggingface.co/datasets/XingjianLi/tomatotest/resolve/main/data/archive_0.tar"],
}

class RGBSemanticDepthDataset(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CustomConfig(name="full", version="1.0.0", description="load both segmentation and depth (for all tar files, 160GB)"),
        CustomConfig(name="sample", version="1.0.0", description="load both segmentation and depth (for 1 tar file, 870MB)"),
        CustomConfig(name="depth", version="1.0.0", description="only load depth (sample)"),
        CustomConfig(name="seg", version="1.0.0", description="only load segmentation (sample)"),
    ]    # Configs initialization
    BUILDER_CONFIG_CLASS = CustomConfig
    def _info(self):
        return DatasetInfo(
            features=Features({
                "left_rgb": datasets.Image(),
                "right_rgb": datasets.Image(),
                "left_semantic": datasets.Image(),
                "left_instance": datasets.Image(),
                "left_depth": datasets.Image(),
                "right_depth": datasets.Image(),
            })
        )
    def _h5_loader(self, bytes_stream, type_dataset):
        # Reference: https://github.com/dwofk/fast-depth/blob/master/dataloaders/dataloader.py#L8-L13
        f = io.BytesIO(bytes_stream)
        h5f = h5py.File(f, "r")
        left_rgb = self._read_jpg(h5f['rgb_left'][:])
        if type_dataset == 'depth':
            right_rgb = self._read_jpg(h5f['rgb_right'][:])
            left_depth = h5f['depth_left'][:].astype(np.float32)
            right_depth = h5f['depth_right'][:].astype(np.float32)
            return left_rgb, right_rgb, np.zeros((1,1)), np.zeros((1,1)), left_depth, right_depth
        elif type_dataset == 'seg':
            left_semantic = h5f['seg_left'][:][:,:,2]
            left_instance = h5f['seg_left'][:][:,:,0] + h5f['seg_left'][:][:,:,1] * 256
            return left_rgb, np.zeros((1,1)), left_semantic, left_instance, np.zeros((1,1)), np.zeros((1,1))
        else:
            right_rgb = self._read_jpg(h5f['rgb_right'][:])
            left_semantic = h5f['seg_left'][:][:,:,2]
            left_instance = h5f['seg_left'][:][:,:,0] + h5f['seg_left'][:][:,:,1] * 256
            left_depth = h5f['depth_left'][:].astype(np.float32)
            right_depth = h5f['depth_right'][:].astype(np.float32)
            return left_rgb, right_rgb, left_semantic, left_instance, left_depth, right_depth
    def _read_jpg(self, bytes_stream):
        return Image.open(io.BytesIO(bytes_stream))
    
    def _split_generators(self, dl_manager):
            # Verifica se um diretório de dados local foi fornecido na configuração
            if self.config.local_data_dir:
                print(f"Carregando dados localmente de: {self.config.local_data_dir}")
                local_data_files = []
                # Itera através da pasta local_data_dir para encontrar todos os arquivos .h5
                for root, _, files in os.walk(self.config.local_data_dir):
                    for file_name in files:
                        if file_name.endswith(".h5"):
                            local_data_files.append(os.path.join(root, file_name))

                # Cria um "gerador de arquivo" para simular o comportamento de dl_manager.iter_archive
                # Isso é necessário porque _generate_examples espera um iterador de (caminho, objeto_arquivo)
                def local_h5_archive_generator(h5_paths_list):
                    for h5_path in h5_paths_list:
                        # O "path" retornado deve ser o nome base do arquivo sem extensão (ex: "image_00000")
                        # para corresponder ao que train.txt/val.txt esperam.
                        filename_no_ext = os.path.basename(h5_path)[:-3] # Remove '.h5'
                        with open(h5_path, "rb") as f:
                            yield filename_no_ext, io.BytesIO(f.read())

                # Separa os arquivos .h5 com base em train.txt e val.txt locais
                # Primeiro, carregue os nomes dos arquivos de train.txt e val.txt
                train_names = set()
                if self.config.local_train_txt and os.path.exists(self.config.local_train_txt):
                    with open(self.config.local_train_txt, 'r', encoding="utf-8") as f:
                        train_names = set(line.strip() for line in f if line.strip())
                else:
                    print(f"AVISO: {self.config.local_train_txt} não encontrado. Tentando baixar.")
                    train_names = set(dl_manager.download(_metadata_urls["train"]).read().decode("utf-8").splitlines())
                    train_names = set(s.strip() for s in train_names if s.strip()) # Clean up

                val_names = set()
                if self.config.local_val_txt and os.path.exists(self.config.local_val_txt):
                    with open(self.config.local_val_txt, 'r', encoding="utf-8") as f:
                        val_names = set(line.strip() for line in f if line.strip())
                else:
                    print(f"AVISO: {self.config.local_val_txt} não encontrado. Tentando baixar.")
                    val_names = set(dl_manager.download(_metadata_urls["val"]).read().decode("utf-8").splitlines())
                    val_names = set(s.strip() for s in val_names if s.strip()) # Clean up


                local_train_h5_paths = [
                    f for f in local_data_files if os.path.basename(f)[:-3] in train_names
                ]
                local_val_h5_paths = [
                    f for f in local_data_files if os.path.basename(f)[:-3] in val_names
                ]

                # Agora, crie os geradores de arquivo para cada split
                local_archives_for_train = [local_h5_archive_generator(local_train_h5_paths)]
                local_archives_for_val = [local_h5_archive_generator(local_val_h5_paths)]

                # Carrega os arquivos train.txt e val.txt locais ou baixa se não existirem
                local_split_metadata = {}
                if self.config.local_train_txt and os.path.exists(self.config.local_train_txt):
                    local_split_metadata["train"] = self.config.local_train_txt
                else:
                    local_split_metadata["train"] = dl_manager.download(_metadata_urls["train"]) # Fallback to download

                if self.config.local_val_txt and os.path.exists(self.config.local_val_txt):
                    local_split_metadata["val"] = self.config.local_val_txt
                else:
                    local_split_metadata["val"] = dl_manager.download(_metadata_urls["val"]) # Fallback to download


                return [
                    SplitGenerator(
                        name=Split.TRAIN,
                        gen_kwargs={
                            "archives": local_archives_for_train,
                            "split_txt": local_split_metadata["train"]
                        },
                    ),
                    SplitGenerator(
                        name=Split.VALIDATION,
                        gen_kwargs={
                            "archives": local_archives_for_val,
                            "split_txt": local_split_metadata["val"]
                        },
                    ),
                ]
            else:
                # Comportamento original: baixar de URLs remotas
                print("Carregando dados remotamente.")
                archives = dl_manager.download_and_extract(_archives_urls)
                split_metadata = dl_manager.download(_metadata_urls)

                return [
                    SplitGenerator(
                        name=Split.TRAIN,
                        gen_kwargs={
                            "archives": [dl_manager.iter_archive(archive) for archive in archives["train"]],
                            "split_txt": split_metadata["train"]
                        },
                    ),
                    SplitGenerator(
                        name=Split.VALIDATION,
                        gen_kwargs={
                            "archives": [dl_manager.iter_archive(archive) for archive in archives["val"]],
                            "split_txt": split_metadata["val"]
                        },
                    ),
                ]

    def _generate_examples(self, archives, split_txt):
        with open(split_txt, encoding="utf-8") as split_f:
            all_splits = split_f.read().split('\n')
            all_splits = [s.strip() for s in all_splits if s.strip()] # Adicione esta linha para limpar
            # print(f"Loaded {len(all_splits)} entries from {split_txt}") # Debugging
        
        for archive_gen in archives: # archives is a list of generators (or iterables)
            for path, file_obj in archive_gen: # file_obj is a BytesIO or similar
                # path é o nome do arquivo sem extensão (ex: "image_00000")
                if path not in all_splits:
                    # print(f"Skipping {path} not in split_txt") # Debugging
                    continue
                # print(f"Adding {path}") # Debugging
                left_rgb, right_rgb, left_semantic, left_instance, left_depth, right_depth = self._h5_loader(file_obj.read(), self.config.dataset_type)
                yield path, { # Use path como ID
                    "left_rgb": left_rgb,
                    "right_rgb": right_rgb,
                    "left_semantic": left_semantic,
                    "left_instance": left_instance,
                    "left_depth": left_depth,
                    "right_depth": right_depth,
                }
    def _get_dataset_filenames(self):
        fs = HfFileSystem()
        all_files = fs.ls("datasets/xingjianli/tomatotest/data")
        filenames = sorted(['/'.join(f['name'].split('/')[-2:]) for f in all_files])
        return filenames