import os
import numpy as np
import time
import torch
import cv2
import json
import matplotlib.pyplot as plt
from utils.draw3d import save_a_image_with_mesh_joints, draw_2d_skeleton, draw_mesh
from utils.read import save_mesh
from utils.warmup_scheduler import adjust_learning_rate
from utils.vis import inv_base_tranmsform, map2uv, base_transform, regist
from utils.zimeval import EvalUtil
from utils.transforms import rigid_align
from LSG.tools.vis import perspective, compute_iou, cnt_area
from LSG.tools.kinematics import mano_to_mpii, MPIIHandJoints
from LSG.tools.registration import registration
import vctoolkit as vc
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, OpenGLPerspectiveCameras, RasterizationSettings, MeshRenderer,
                                MeshRasterizer, HardPhongShader, TexturesUV, FoVPerspectiveCameras, PointLights,
                                SoftPhongShader, TexturesVertex)
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

    def run(self):
        if self.args.PHASE == 'train':
            if self.val_loader is not None and self.epoch > 0:
                self.best_val_loss = self.eval()
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                self.epoch = epoch
                t = time.time()
                if self.args.world_size > 1:
                    self.train_loader.sampler.set_epoch(epoch)
                train_loss = self.train()
                t_duration = time.time() - t
                if self.val_loader is not None:
                    val_loss = self.eval()
                else:
                    val_loss = np.float('inf')

                info = {
                    'current_epoch': self.epoch,
                    'epochs': self.max_epochs,
                    'train_loss': train_loss,
                    'test_loss': val_loss,
                    't_duration': t_duration
                }

                self.writer.print_info(info)
                # if val_loss < self.best_val_loss:
                self.writer.save_checkpoint(self.model, self.optimizer, None, self.epoch)
                    # self.best_test_loss = val_loss
                #self.writer.save_checkpoint(self.model, self.optimizer, None, self.epoch)
                self.pred()
        elif self.args.PHASE == 'eval':
            self.eval()
        elif self.args.PHASE == 'pred':
            self.pred()
        elif self.args.PHASE == 'demo':
            self.demo()
        elif self.args.PHASE == 'demo_test_new_data':
            self.demo_test_new_data()
        elif self.args.PHASE == 'demo_pt3D':
            self.demo_pt3D()
        else:
            raise Exception('PHASE ERROR')

    def phrase_data(self, data):
        for key, val in data.items():
            try:
                if isinstance(val, list):
                    data[key] = [d.to(self.device) for d in data[key]]
                else:
                    data[key] = data[key].to(self.device)
            except:
                pass
        return data

    def board_scalar(self, phase, n_iter, lr=None, **kwargs):
        split = '/'
        for key, val in kwargs.items():
            if 'loss' in key:
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.board.add_scalar(phase + split + key, val, n_iter)
            if lr:
                self.board.add_scalar(phase + split + 'lr', lr, n_iter)

    def draw_results(self, data, out, loss, batch_id, aligned_verts=None):
        img_cv2 = inv_base_tranmsform(data['img'][batch_id].cpu().numpy())[..., :3]
        draw_list = []
        draw_list.append(img_cv2.copy())
        if 'joint_img' in data:
            draw_list.append( vc.render_bones_from_uv(np.flip(data['joint_img'][batch_id, :, :2].cpu().numpy()*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                      img_cv2.copy(), MPIIHandJoints, thickness=2) )
        if 'joint_img' in out:
            try:
                draw_list.append( vc.render_bones_from_uv(np.flip(out['joint_img'][batch_id, :, :2].detach().cpu().numpy()*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                         img_cv2.copy(), MPIIHandJoints, thickness=2) )
            except:
                draw_list.append(img_cv2.copy())
        if 'root' in data:
            root = data['root'][batch_id:batch_id+1, :3]
        else:
            root = torch.FloatTensor([[0, 0, 0.6]]).to(data['img'].device)
        if 'verts' in data:
            vis_verts_gt = img_cv2.copy()
            verts = data['verts'][batch_id:batch_id+1, :, :3] * 0.2 + root
            vp = perspective(verts.permute(0, 2, 1), data['calib'][batch_id:batch_id+1, :4])[0].cpu().numpy().T
            for i in range(vp.shape[0]):
                cv2.circle(vis_verts_gt, (int(vp[i, 0]), int(vp[i, 1])), 1, (255, 0, 0), -1)
            draw_list.append(vis_verts_gt)
        if 'verts' in out:
            try:
                vis_verts_pred = img_cv2.copy()
                if aligned_verts is None:
                    verts = out['verts'][batch_id:batch_id+1, :, :3] * 0.2 + root
                else:
                    verts = aligned_verts
                vp = perspective(verts.permute(0, 2, 1), data['calib'][batch_id:batch_id+1, :4])[0].detach().cpu().numpy().T
                for i in range(vp.shape[0]):
                    cv2.circle(vis_verts_pred, (int(vp[i, 0]), int(vp[i, 1])), 1, (255, 0, 0), -1)
                draw_list.append(vis_verts_pred)
            except:
                draw_list.append(img_cv2.copy())

        return np.concatenate(draw_list, 1)

    def board_img(self, phase, n_iter, data, out, loss, batch_id=0):
        draw = self.draw_results(data, out, loss, batch_id)
        self.board.add_image(phase + '/res', draw.transpose(2, 0, 1), n_iter)

    def train(self):
        self.writer.print_str('TRAINING ..., Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.train()
        total_loss = 0
        forward_time = 0.
        backward_time = 0.
        start_time = time.time()
        for step, data in enumerate(self.train_loader):
            ts = time.time()
            adjust_learning_rate(self.optimizer, self.epoch, step, len(self.train_loader), self.cfg.TRAIN.LR, self.cfg.TRAIN.LR_DECAY, self.cfg.TRAIN.DECAY_STEP, self.cfg.TRAIN.WARMUP_EPOCHS)
            data = self.phrase_data(data)
            self.optimizer.zero_grad()
            out = self.model(data['img'])
            tf = time.time()
            forward_time += tf - ts
            losses = self.loss(verts_pred=out.get('verts'),
                               verts_rough_pred=out.get('verts_rough'),
                               joint_img_pred=out['joint_img'],
                               verts_gt=data.get('verts'),
                               joint_img_gt=data['joint_img'],
                               face=self.face,
                               aug_param=(None, data.get('aug_param'))[self.epoch>4],
                               bb2img_trans=data.get('bb2img_trans'),
                               size=data['img'].size(2),
                               mask_gt=data.get('mask'),
                               trans_pred=out.get('trans'),
                               alpha_pred=out.get('alpha'),
                               img=data.get('img'))
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()
            tb = time.time()
            backward_time +=  tb - tf

            self.total_step += 1
            total_loss += loss.item()
            if self.board is not None:
                self.board_scalar('train', self.total_step, self.optimizer.param_groups[0]['lr'], **losses)
            if self.total_step % 100 == 0:
                cur_time = time.time()
                duration = cur_time - start_time
                start_time = cur_time
                info = {
                    'train_loss': loss.item(),
                    'l1_loss': losses.get('verts_loss', 0),
                    'epoch': self.epoch,
                    'max_epoch': self.max_epochs,
                    'step': step,
                    'max_step': len(self.train_loader),
                    'total_step': self.total_step,
                    'step_duration': duration,
                    'forward_duration': forward_time,
                    'backward_duration': backward_time,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.writer.print_step_ft(info)
                forward_time = 0.
                backward_time = 0.

        if self.board is not None:
            self.board_img('train', self.epoch, data, out, losses)

        return total_loss / len(self.train_loader)

    def eval(self):
        self.writer.print_str('EVALING ... Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.eval()
        evaluator_2d = EvalUtil()
        evaluator_rel = EvalUtil()
        evaluator_pa = EvalUtil()
        mask_iou = []
        joint_cam_errors = []
        pa_joint_cam_errors = []
        joint_img_errors = []
        with torch.no_grad():
            for step, data in enumerate(self.val_loader):
                if self.board is None and step % 100 == 0:
                    print(step, len(self.val_loader))
                # get data then infernce
                data = self.phrase_data(data)
                out = self.model(data['img'])

                # get vertex pred
                verts_pred = out['verts'][0].cpu().numpy() * 0.2
                joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, verts_pred)) * 1000.0

                # get mask pred
                mask_pred = out.get('mask')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (data['img'].size(3), data['img'].size(2)))
                else:
                    mask_pred = np.zeros((data['img'].size(3), data['img'].size(2)), np.uint8)

                # get uv pred
                joint_img_pred = out.get('joint_img')
                if joint_img_pred is not None:
                    joint_img_pred = joint_img_pred[0].cpu().numpy() * data['img'].size(2)
                else:
                    joint_img_pred = np.zeros((21, 2), dtype=np.float)

                # pck
                joint_cam_gt = data['joint_cam'][0].cpu().numpy() * 1000.0
                joint_cam_align = rigid_align(joint_cam_pred, joint_cam_gt)
                evaluator_2d.feed(data['joint_img'][0].cpu().numpy() * data['img'].size(2), joint_img_pred)
                evaluator_rel.feed(joint_cam_gt, joint_cam_pred)
                evaluator_pa.feed(joint_cam_gt, joint_cam_align)

                # error
                if 'mask_gt' in data.keys():
                    mask_iou.append(compute_iou(mask_pred, cv2.resize(data['mask_gt'][0].cpu().numpy(), (data['img'].size(3), data['img'].size(2)))))
                else:
                    mask_iou.append(0)
                joint_cam_errors.append(np.sqrt(np.sum((joint_cam_pred - joint_cam_gt) ** 2, axis=1)))
                pa_joint_cam_errors.append(np.sqrt(np.sum((joint_cam_gt - joint_cam_align) ** 2, axis=1)))
                joint_img_errors.append(np.sqrt(np.sum((data['joint_img'][0].cpu().numpy()*data['img'].size(2) - joint_img_pred) ** 2, axis=1)))

            # get auc
            _1, _2, _3, auc_rel, pck_curve_rel, thresholds2050 = evaluator_rel.get_measures(20, 50, 20)
            _1, _2, _3, auc_pa, pck_curve_pa, _ = evaluator_pa.get_measures(20, 50, 20)
            _1, _2, _3, auc_2d, pck_curve_2d, _ = evaluator_2d.get_measures(0, 30, 20)
            # get error
            miou = np.array(mask_iou).mean()
            mpjpe = np.array(joint_cam_errors).mean()
            pampjpe = np.array(pa_joint_cam_errors).mean()
            uve = np.array(joint_img_errors).mean()

            if self.board is not None:
                self.board_scalar('test', self.epoch, **{'auc_loss': auc_rel, 'pa_auc_loss': auc_pa, '2d_auc_loss': auc_2d, 'mIoU_loss': miou, 'uve': uve, 'mpjpe_loss': mpjpe, 'pampjpe_loss': pampjpe})
                self.board_img('test', self.epoch, data, out, {})
            elif self.args.world_size < 2:
                print( f'pampjpe: {pampjpe}, mpjpe: {mpjpe}, uve: {uve}, miou: {miou}, auc_rel: {auc_rel}, auc_pa: {auc_pa}, auc_2d: {auc_2d}')
                print('thresholds2050', thresholds2050)
                print('pck_curve_all_pa', pck_curve_pa)
            self.writer.print_str( f'pampjpe: {pampjpe}, mpjpe: {mpjpe}, uve: {uve}, miou: {miou}, auc_rel: {auc_rel}, auc_pa: {auc_pa}, auc_2d: {auc_2d}')

        return pampjpe

    def pred(self):
        self.writer.print_str('PREDICING ... Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.eval()
        xyz_pred_list, verts_pred_list = list(), list()
        with torch.no_grad():
            for step, data in enumerate(self.test_loader):
                if self.board is None and step % 100 == 0:
                    print(step, len(self.test_loader))
                data = self.phrase_data(data)
                out = self.model(data['img'])
                # get verts pred
                verts_pred = out['verts'][0].cpu().numpy() * 0.2

                # get mask pred
                mask_pred = out.get('mask')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (data['img'].size(3), data['img'].size(2)))
                    try:
                        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours.sort(key=cnt_area, reverse=True)
                        poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                    except:
                        poly = None
                else:
                    poly = None

                # get uv pred
                joint_img_pred = out.get('joint_img')
                if joint_img_pred is not None:
                    joint_img_pred = joint_img_pred[0].cpu().numpy() * data['img'].size(2)
                    verts_pred, align_state = registration(verts_pred, joint_img_pred, self.j_reg, data['calib'][0].cpu().numpy(), self.cfg.DATA.SIZE, poly=poly)

                # get joint_cam
                joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, verts_pred))

                # track data
                xyz_pred_list.append(joint_cam_pred)
                verts_pred_list.append(verts_pred)
                if self.cfg.TEST.SAVE_PRED:
                    save_a_image_with_mesh_joints(inv_base_tranmsform(data['img'][0].cpu().numpy())[:, :, ::-1],
                                                  mask_pred, poly, data['calib'][0, 0:3, 0:3].cpu().numpy(), verts_pred,
                                                  self.npface,
                                                  joint_img_pred, joint_cam_pred,
                                                  os.path.join(r"C:\Users\29785\Desktop\time_now", 'rough_GE_pic',
                                                               "rough" + 'GE' + str(step) + '_plot.jpg'))
                    save_mesh(os.path.join(r"C:\Users\29785\Desktop\time_now\rough_GE_mesh", str(step)+ 'rough_GE_mesh.ply'), verts_pred, self.npface)
                    draw = self.draw_results(data, out, {}, 0, aligned_verts=torch.from_numpy(verts_pred).float()[None, ...])[..., ::-1]
                    cv2.imwrite(os.path.join(self.args.out_dir, self.cfg.TEST.SAVE_DIR, f'{step}.png'), draw)

        # dump results
        xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        verts_pred_list = [x.tolist() for x in verts_pred_list]
        # save to a json
        with open(os.path.join(self.args.out_dir, f'{self.args.exp_name}{self.epoch}.json'), 'w') as fo:
            json.dump(
                [
                    xyz_pred_list,
                    verts_pred_list
                ], fo)
        self.writer.print_str('Dumped %d joints and %d verts predictions to %s' % (
            len(xyz_pred_list), len(verts_pred_list), os.path.join(self.args.work_dir, 'out', self.args.exp_name, f'{self.args.exp_name}.json')))
        if self.args.Local_testing:

            import subprocess
            input_dir = r'E:\LiteSpiralGCN\LSG\out\MultipleDatasets\LiteSpiralGCN' #Enter your data input directory
            output_dir = r'C:\Users\29785\Desktop\ok\LiteSpiralGCN_GCN_MHA_TWO_multidata' #Output directory for results
            pred_file_name =r'LiteSpiralGCN{}.json'.format(self.epoch)   #Adjust based on the experiment name:experiment name{}.json
            command = ['python', r'LSG\freihand-master\eval.py', input_dir, output_dir,  #
                       '--pred_file_name', pred_file_name]

            subprocess.run(command)

    def demo(self):
        self.model.eval()
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Hand Mesh Estimation", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("Hand Mesh Estimation", 500, 500)
        cv2.namedWindow("Hand pose Estimation", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("Hand pose Estimation", 500, 500)
        cv2.namedWindow("cropped_image", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("cropped_image", 500, 500)
        hand_mesh_model = self.model
        K=np.array([[730, 0, 160],
                [0,740, 160],
                [0, 0, 1]])
        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_height, original_width = frame_rgb.shape[:2]
            target_size = 256
            crop_x = (original_width - target_size) // 2
            crop_y = (original_height - target_size) // 2
            if not ret:
                break
            with torch.no_grad():
                    cropped_image = frame_rgb[crop_y:crop_y + target_size, crop_x:crop_x + target_size]
                    input = torch.from_numpy(base_transform(cropped_image, size=128)).unsqueeze(0).to(self.device)
                    out = self.model(input)
                    vertex = (out['verts'][0].cpu() * 0.2).numpy()
                    uv_pred = out['joint_img']
                    uv_point_pred = (uv_pred * target_size).cpu().numpy()
                    vertex, align_state = regist(vertex, uv_point_pred[0], self.j_reg, K, target_size
                                                       )
                    skeleton_overlay = draw_2d_skeleton(cropped_image[..., ::-1],uv_point_pred[0])
                    rend_img_overlay = draw_mesh(cropped_image[..., ::-1], K, vertex, self.npface)

                    cv2.imshow("Hand pose Estimation",skeleton_overlay )
                    cv2.imshow("Hand Mesh Estimation", rend_img_overlay )
                    cv2.imshow("cropped_image", cropped_image[..., ::-1])


                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

    def demo_test_new_data(self):

        self.model.eval()
        hand_mesh_model = self.model
        K=np.array([[617.061, 0, 194.564],
                [0,617.061, 196.353],
                [0, 0, 1]])
        # Ensure the output folder exists
        output_folder = r"C:\Users\29785\Desktop\GEtest\image_pose_alignedmesh_GE"  #The storage path for the prediction results generated by the model.
        os.makedirs(output_folder, exist_ok=True)

        image_folder_path = r"data\Ge\images" #The path to the hand image to be inputted into the model.
        # Get the list of image files in the folder
        image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        cv2.namedWindow("Hand Mesh Estimation", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("Hand Mesh Estimation", 500, 500)
        cv2.namedWindow("Hand pose Estimation", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("Hand pose Estimation", 500, 500)
        cv2.namedWindow("cropped_image", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("cropped_image", 500, 500)

        for image_file in image_files:
            image_path = os.path.join(image_folder_path, image_file)
            frame = cv2.imread(image_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_height, original_width = frame_rgb.shape[:2]
            target_size = 256
            crop_x = (original_width - target_size) // 2
            crop_y = (original_height - target_size) // 2
            with torch.no_grad():
                    cropped_image = frame_rgb[crop_y:crop_y + target_size, crop_x:crop_x + target_size]
                    t0 = time.time()
                    input = torch.from_numpy(base_transform(cropped_image, size=128)).unsqueeze(0).to(self.device)
                    out = self.model(input)
                    vertex = (out['verts'][0].cpu() * 0.2).numpy()
                    uv_pred = out['joint_img']
                    uv_point_pred = (uv_pred * target_size).cpu().numpy()
                    vertex, align_state = regist(vertex, uv_point_pred[0], self.j_reg, K, target_size)
                    save_mesh(
                        os.path.join(r"C:\Users\29785\Desktop\GEtest\mesh_GE", str(image_file) + '_mesh.ply'),  #The storage path for the hand mesh files (.ply).
                        vertex, self.npface)
                    skeleton_overlay = draw_2d_skeleton(cropped_image[..., ::-1],uv_point_pred[0])
                    rend_img_overlay = draw_mesh(cropped_image[..., ::-1], K, vertex, self.npface)
                    cv2.imshow("Hand pose Estimation",skeleton_overlay )
                    cv2.imshow("Hand Mesh Estimation", rend_img_overlay )
                    cv2.imshow("cropped_image", cropped_image[..., ::-1])

                    c=cropped_image[..., ::-1]

                    combined_image = np.concatenate((cropped_image[..., ::-1], skeleton_overlay, rend_img_overlay[:, :, :3]), axis=1)
                    output_path = os.path.join(output_folder, f"result_{image_file}")
                    cv2.imwrite(output_path,combined_image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cv2.destroyAllWindows()

    def demo_pt3D(self):

        self.model.eval()

        R, T = look_at_view_transform(dist=2, elev=180, azim=0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=20)
        raster_settings = RasterizationSettings(image_size=500)
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
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Hand pose Estimation", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("Hand pose Estimation", 500, 500)
        cv2.namedWindow("cropped_image", cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow("cropped_image", 500, 500)
        hand_mesh_model = self.model
        K = np.array([[730, 0, 160],
                      [0, 740, 160],
                      [0, 0, 1]])
        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_height, original_width = frame_rgb.shape[:2]
            target_size = 256
            crop_x = (original_width - target_size) // 2
            crop_y = (original_height - target_size) // 2
            if not ret:
                break
            with torch.no_grad():
                cropped_image = frame_rgb[crop_y:crop_y + target_size, crop_x:crop_x + target_size]
                input = torch.from_numpy(base_transform(cropped_image, size=128)).unsqueeze(0).to(self.device)
                # print(input.shape)

                out = self.model(input)
                vertex = (out['verts'][0].cpu() * 0.2).numpy()
                uv_pred = out['joint_img']
                uv_point_pred = (uv_pred * target_size).cpu().numpy()
                vertex, align_state = regist(vertex, uv_point_pred[0], self.j_reg, K, target_size
                                             )
                skeleton_overlay = draw_2d_skeleton(cropped_image[..., ::-1], uv_point_pred[0])
                vertex = vertex.astype(np.float32)
                faces = self.npface.astype(np.int64)
                verts_tensor = torch.from_numpy(vertex).unsqueeze(0).to(self.device) * 10

                faces_tensor = torch.from_numpy(faces).unsqueeze(0).to(self.device)
                orange_color = torch.tensor([0.694, 0.761, 0.941], device=self.device)
                verts_rgb = orange_color.expand(verts_tensor.shape[0], verts_tensor.shape[1], 3)  # (N, V, 3)
                textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
                mesh = Meshes(
                    verts=verts_tensor,
                    faces=faces_tensor,
                    textures=textures
                )
                rendered_image = renderer(mesh)
                cv2.imshow("Hand pose Estimation", skeleton_overlay)
                cv2.imshow("cropped_image", cropped_image[..., ::-1])
                rendered_image_numpy = rendered_image[0, ..., :3].cpu().numpy()
                rendered_image_numpy = (rendered_image_numpy * 255).astype(np.uint8)
                rendered_image_numpy = cv2.flip(rendered_image_numpy, -1)
                cv2.imshow("PyTorch3D Mesh Rendering", rendered_image_numpy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()





