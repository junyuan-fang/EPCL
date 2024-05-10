class Config:
    def __init__(self):
        self.model = 'net_epcl'
        self.arch = 'epcl'
        self.intraLayer = 'PointMixerIntraSetLayer'
        self.interLayer = 'PointMixerInterSetLayer'
        self.transdown = 'SymmetricTransitionDownBlock'
        self.transup = 'SymmetricTransitionUpBlock'
        self.nsample = [8, 16, 16, 16, 16]
        self.downsample = [1, 4, 4, 4, 4]
        self.drop_rate = 0.1
        self.fea_dim = 6
        self.classes = 13
        self.cudnn_benchmark = False
        self.train_batch = 2
        self.val_batch = 4
        self.test_batch = 10
        self.train_worker = 4
        self.val_worker = 4
        self.dataset = None
        self.scannet_train_root = None
        self.scannet_test_root = None
        self.scannet_semgseg_root = None
        self.shapenet_root = None
        self.shapenetcore_root = None
        self.s3dis_root = None
        #scnene_path
        self.arkit_train_root = "/scratch/project_2002051/junyuan/cvpr24-challenge/data/ChallengeDevelopmentSet"
        self.arkit_test_root = "/scratch/project_2002051/junyuan/cvpr24-challenge/data/ChallengeTestSet"
        #query_path
        self.development_query_root = "/scratch/project_2002051/junyuan/cvpr24-challenge/challenge/benchmark_file_lists/queries_development_scenes.csv"
        self.test_query_root = "/scratch/project_2002051/junyuan/cvpr24-challenge/challenge/benchmark_file_lists/queries_test_scenes.csv"
        #mask_path
        self.development_mask_root = "/scratch/project_2002051/junyuan/cvpr24-challenge/challenge/benchmark_data/gt_development_scenes"
        self.test_mask_root = ""
        
        self.loop = 30
        self.ignore_label = 255
        self.test_area = 5
        self.train_voxel_max = 40000
        self.eval_voxel_max = 800000
        self.voxel_size = 0.04
        self.mode_train = 'train'
        self.mode_eval = 'val'
        self.aug = 'pointtransformer'
        self.crop_npart = 0
        self.optim = 'Adam'
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.lr_STEP_SIZE = 6
        self.lr_GAMMA = 0.1
        self.schedule = [0.6, 0.8]
        self.AMP_LEVEL = 'O0'
        self.PRECISION = 32
        self.load_model = None
        self.resume = False
        self.on_train = True
        self.shell = 'run_old.sh'
        self.MYCHECKPOINT = './checkpoints'
        self.computer = None
        self.NUM_GPUS = 1
        self.epochs = 10
        self.CHECKPOINT_PERIOD = 1
        self.strict_load = True
        self.MASTER_ADDR = 'localhost'
        self.MASTER_PORT = '29500'
        self.on_neptune = False
        self.off_text_logger = True
        self.neptune_proj = "junyuan-fang/EPCL"
        self.print_freq = 1
        self.neptune_id = "junyuan-fang"
        self.neptune_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMDRjNjUwMC0wYmJmLTQ5MjUtYmI1Mi0xNThhNWIwNmVlOTUifQ"
        self.load_model = "/home/fangj1/Code/Vision-Language-on-3D-Scene-Understanding/EPCL/indoor_segmentation/checkpoints/epoch=062--mIoU_val=0.6972--.ckpt"