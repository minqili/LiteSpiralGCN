import os
import numpy as np
import time
import torch
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex,
    PointLights
)
from torch.backends import cudnn

from LSG.configs.config import get_cfg
from utils.draw3d import draw_2d_skeleton
from utils.vis import base_transform, regist
import os.path as osp
from LSG.build import build_model, build_dataset
from LSG.configs.config import get_cfg
from options.cfg_options import CFGOptions
from LSG.runner import Runner
import os.path as osp
from utils import utils
from utils.writer import Writer
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def main(args):
    # get config
    cfg = setup(args)

    # device
    args.rank = 0
    args.world_size = 1
    args.n_threads = 4
    if -1 in cfg.TRAIN.GPU_ID or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('CPU mode')
    elif len(cfg.TRAIN.GPU_ID) == 1:
        device = torch.device('cuda', cfg.TRAIN.GPU_ID[0])
        print('CUDA ' + str(cfg.TRAIN.GPU_ID) + ' Used')
    else:
        raise Exception('Do not support multi-GPU training')
    cudnn.benchmark = True
    cudnn.deterministic = False  #FIXME

    # print config
    if args.rank == 0:
        print(cfg)
        print(args.exp_name)
    exec('from LSG.models.{} import {}'.format(cfg.MODEL.NAME.lower(), cfg.MODEL.NAME))
    exec('from LSG.datasets.{} import {}'.format(cfg.TRAIN.DATASET.lower(), cfg.TRAIN.DATASET))
    exec('from LSG.datasets.{} import {}'.format(cfg.VAL.DATASET.lower(), cfg.VAL.DATASET))

    # dir
    args.work_dir = osp.dirname(osp.realpath(__file__))
    args.out_dir = osp.join(args.work_dir, 'out', cfg.TRAIN.DATASET, args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
    args.board_dir = osp.join(args.out_dir, 'board')
    args.eval_dir = osp.join(args.out_dir, cfg.VAL.SAVE_DIR)
    args.test_dir = osp.join(args.out_dir, cfg.TEST.SAVE_DIR)
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        os.makedirs(args.board_dir, exist_ok=True)
        os.makedirs(args.eval_dir, exist_ok=True)
        os.makedirs(args.test_dir, exist_ok=True)
    except: pass

    # log
    writer = Writer(args)
    writer.print_str(args)
    writer.print_str(cfg)
    board = SummaryWriter(args.board_dir) if cfg.PHASE == 'train' and args.rank == 0 else None

    # model
    model = build_model(cfg).to(device)

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # resume
    if cfg.MODEL.RESUME:
        if len(cfg.MODEL.RESUME.split('/')) > 1:
            model_path = cfg.MODEL.RESUME
        else:
            model_path = osp.join(args.checkpoints_dir, cfg.MODEL.RESUME)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        writer.print_str('Resume from: {}, start epoch: {}'.format(model_path, epoch))
        print('Resume from: {}, start epoch: {}'.format(model_path, epoch))
    else:
        epoch = 0
        writer.print_str('Train from 0 epoch')

    # data
    kwargs = {"pin_memory":False , "num_workers": 8, "drop_last": False}
    if args.PHASE in ['train',]:
        train_dataset = build_dataset(cfg, 'train', writer=writer)
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler, **kwargs)
    else:
        print('Need not trainloader')
        train_loader = None

    if args.PHASE in ['train', 'eval','pred']:
        eval_dataset = build_dataset(cfg, 'val', writer=writer)
        eval_sampler = None
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False, sampler=eval_sampler, **kwargs)
    else:
        print('Need not eval_loader')
        eval_loader = None

    if args.PHASE in ['train', 'pred']:
        test_dataset = build_dataset(cfg, 'test', writer=writer)
        test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, **kwargs)
    else:
        print('Need not testloader')
        test_loader = None

    if args.PHASE in ['demo']:
        print('Need not any loader')
    if args.PHASE in ['demo_test_new_data']:
        print('Need not any loader')
    if args.PHASE in ['demo_pt3D']:
        print('Need not any loader')
    # run
    runner = Runner(cfg, args, model, train_loader, eval_loader, test_loader, optimizer, writer, device, board, start_epoch=epoch)
    runner.run()
class Runner(object):
    def __init__(self, cfg, args, model, train_loader, val_loader, test_loader, optimizer, writer, device, board, start_epoch=0):
        super(Runner, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        face = np.load(os.path.join(cfg.MODEL.MANO_PATH, 'right_faces.npy'))
        self.npface = face
        self.face = torch.from_numpy(face).long()
        self.j_reg = np.load(os.path.join(self.cfg.MODEL.MANO_PATH, 'j_reg.npy'))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = cfg.TRAIN.EPOCHS
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.board = board
        self.start_epoch = start_epoch
        self.epoch = max(start_epoch - 1, 0)
        if self.args.PHASE == 'train':
            self.total_step = self.start_epoch * (len(self.train_loader.dataset) // cfg.TRAIN.BATCH_SIZE)
            try:
                self.loss = self.model.loss
            except:
                self.loss = self.model.module.loss
        self.best_val_loss = np.float('inf')
        print('runner init done')
    def demo_pt3D(self):

        self.model.eval()
        # 初始化 PyTorch3D 渲染器
        R, T = look_at_view_transform(dist=2, elev=180, azim=0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T,fov=20)
        raster_settings = RasterizationSettings(image_size=700)
        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        # 初始化摄像头

        cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
        # 获取并打印当前帧速率
        current_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"当前帧速率: {current_fps}")

        # 设置所需的帧速率
        desired_fps = 60  # 设置为所需的帧速率

        # 尝试设置摄像头的帧速率
        cap.set(cv2.CAP_PROP_FPS, desired_fps)

        # 获取并验证设置后的帧速率
        new_fps = cap.get(cv2.CAP_PROP_FPS)
        if new_fps == desired_fps:
            print(f"帧速率已设置为 {desired_fps} FPS")
        else:
            print(f"无法设置所需的帧速率，当前帧速率为 {new_fps} FPS")
        # 创建窗口并设置属性
        #cv2.namedWindow("Hand Mesh Estimation", cv2.WND_PROP_FULLSCREEN)
        #cv2.resizeWindow("Hand Mesh Estimation", 500, 500)
        cv2.namedWindow("Hand pose Estimation", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("Hand pose Estimation", 700, 700)
        cv2.namedWindow("cropped_image", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("cropped_image", 700, 700)

        # 加载手部姿势估计模型和手部网格估计模型
        hand_mesh_model = self.model  # 请替换为您的手部网格估计模型
        K=np.array([[730, 0, 160],
                [0,740, 160],
                [0, 0, 1]])
        frame_count = 0
        start_time = time.time()

        while True:
            # 从摄像头捕获一帧
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 获取原始图像的高度和宽度
            original_height, original_width = frame_rgb.shape[:2]

            # 设置剪裁后的目标尺寸
            target_size = 256

            # 计算剪裁区域的左上角坐标
            crop_x = (original_width - target_size) // 2
            crop_y = (original_height - target_size) // 2
            if not ret:
                break
            with torch.no_grad():


                    # 剪裁图像
                    cropped_image = frame_rgb[crop_y:crop_y + target_size, crop_x:crop_x + target_size]
                    # print(cropped_image.shape)
                    # print(image.shape)
                    # image = cv2.resize(image, (480, 480))
                    # print(image.shape)
                    input = torch.from_numpy(base_transform(cropped_image, size=128)).unsqueeze(0).to(self.device)
                    #print(input.shape)


                    out = self.model(input)
                    # silhouette
                    #mask_pred = out.get('mask_pred')
                    # if mask_pred is not None:
                    #     mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    #     mask_pred = cv2.resize(mask_pred, (cropped_image.size(3), cropped_image.size(2)))
                    #     try:
                    #         contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    #         contours.sort(key=cnt_area, reverse=True)
                    #         poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                    #     except:
                    #         poly = None
                    # else:
                    #     #mask_pred = np.zeros([cropped_image.size(3), cropped_image.size(2)])
                    #poly = None
                    # vertex
                    # pred = out['verts']
                    vertex = (out['verts'][0].cpu()*0.2).numpy()
                    uv_pred = out['joint_img']
                    # if uv_pred.ndim == 4:
                    #     uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (image.size(2), image.size(3)))
                    # else:
                    uv_point_pred = (uv_pred * target_size).cpu().numpy()
                    vertex, align_state = regist(vertex, uv_point_pred[0], self.j_reg, K, target_size
                                                       )
                    skeleton_overlay = draw_2d_skeleton(cropped_image[..., ::-1],uv_point_pred[0])
                    #rend_img_overlay = draw_mesh_opencv(cropped_image[..., ::-1], K, vertex, self.npface)
                    # 将模型输出转换为 PyTorch3D 的网格格式
                    # 假设 vertex 是一个 numpy 数组，将它转换为 float32 类型
                    vertex = vertex.astype(np.float32)

                    # 假设 self.npface 是一个 numpy 数组，将它转换为 int64 类型（面的张量应该是 LongTensor）
                    faces = self.npface.astype(np.int64)

                    # 将 numpy 数组转换为 torch 张量
                    verts_tensor = torch.from_numpy(vertex).unsqueeze(0).to(self.device)*10

                    faces_tensor = torch.from_numpy(faces).unsqueeze(0).to(self.device)
                    # 创建橘黄色纹理的RGB值
                    orange_color = torch.tensor([0.694, 0.761,0.941], device=self.device)

                    # 将顶点颜色设置为橘黄色
                    verts_rgb = orange_color.expand(verts_tensor.shape[0], verts_tensor.shape[1], 3)  # (N, V, 3)

                    # 创建橘黄色的顶点纹理
                    textures = TexturesVertex(verts_features=verts_rgb.to(self.device))

                    # 为 PyTorch3D 创建网格
                    # 创建包含纹理的网格

                    mesh = Meshes(
                        verts=verts_tensor,
                        faces=faces_tensor,
                        textures=textures
                    )
                    # print("Verts tensor type:", verts_tensor.dtype)  # 应该是 torch.float32
                    # print("Faces tensor type:", faces_tensor.dtype)  # 应该是 torch.int64
                    # # 现在使用 PyTorch3D 渲染器来渲染网格
                    rendered_image = renderer(mesh)

                    # 在屏幕上实时显示
                    cv2.imshow("Hand pose Estimation",skeleton_overlay )
                    #cv2.imshow("Hand Mesh Estimation", rend_img_overlay )
                    cv2.imshow("cropped_image", cropped_image[..., ::-1])
                    # 将 PyTorch3D 渲染的图像转换为 OpenCV 可以显示的格式
                    rendered_image_numpy = rendered_image[0, ..., :3].cpu().numpy()

                    rendered_image_numpy = (rendered_image_numpy * 255).astype(np.uint8)
                    # 假设 `rendered_image_numpy` 是渲染后的图像
                    rendered_image_numpy = cv2.flip(rendered_image_numpy, -1)  # 0 表示沿着x轴翻转

                    cv2.imshow("PyTorch3D Mesh Rendering", rendered_image_numpy)
                    # 更新帧数
                    frame_count += 1

                    # 计算已经过的时间
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    # 如果已经过了1秒，计算FPS
                    if elapsed_time >= 1.0:
                        fps = (frame_count / elapsed_time)
                        print(f"FPS: {fps:.2f}")

                        # 重置计时器
                        frame_count = 0
                        start_time = current_time

                    # 检测按键，按下'q'键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # 释放摄像头和关闭窗口
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    abc=Runner()
