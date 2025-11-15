import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import numpy
import random
import copy
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
import datasets
from torchvision import transforms
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats , split_class_dataset
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate ,load
from logger import make_logger
from data import FixTransform

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name'],cfg['d_mode']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    print(cfg['gm'])
    #server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset = fetch_dataset(cfg['data_name'])
    # print(len(server_dataset['train'].data))
    # print(len(client_dataset['train'].data))
    # for i in range(2):
    #     print(server_dataset['train'][i])
    process_dataset(client_dataset)
    #server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
                                                                                        #    client_dataset['train'])
    # print(len(server_dataset['train'].data))
    # print(len(client_dataset['train'].data))
    #data_loader = make_data_loader(server_dataset, 'global')
    data_loader = make_data_loader(client_dataset, 'global')
    # print(cfg)
    # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    if cfg['world_size']==1:
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    elif cfg['world_size']>1:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = eval('models.{}()'.format(cfg['model_name']))
        model = torch.nn.DataParallel(model,device_ids = [0, 1])
        model.to(cfg["device"])
    # print(model)
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    # if cfg['sbn'] == 1:
    #     batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    # elif cfg['sbn'] == 0:
    #     batchnorm_dataset = client_dataset['train']
    # else:
    #     raise ValueError('Not valid sbn')
    # print(len(batchnorm_dataset))
    batchnorm_dataset = client_dataset['train']
    # data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    # data_split = split_class_dataset(client_dataset,cfg['data_split_mode'])
    if cfg['d_mode'] == 'old':
        data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    elif cfg['d_mode'] == 'new':
        data_split = split_class_dataset(client_dataset,cfg['data_split_mode'])
    if cfg['loss_mode'] != 'sup':
        metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                         'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # if cfg['loss_mode'] == 'sim':
    #     metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # print(metric.metric_name['train'])
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'],load_tag='best')
        # model_tag = '0_CIFAR10_resnet9_50000_sup_100_0.1_iid_5-5_0.2_1'
        # # load_tag = 'checkpoint'
        # load_tag = 'best'
        # path ='/home/sampathkoti/codes/SemiFL/output/model/'
        # if os.path.exists('{}/{}_{}.pt'.format(path,model_tag, load_tag)):
        #     result = load('{}/{}_{}.pt'.format(path,model_tag, load_tag))
        # else:
        #     print('Not exists model tag: {}, start from scratch'.format(model_tag))
        #     from datetime import datetime
        #     from logger import Logger
        #     last_epoch = 1
        #     logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        #     logger = Logger(logger_path)
        #     result = {'epoch': last_epoch, 'logger': logger}
        # if True:
        #     print('Resume from {}'.format(result['epoch']))
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            # supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            supervised_clients = result['supervised_clients']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        # if last_epoch > 1:
        #     model.load_state_dict(result['model_state_dict'])
        #     optimizer.load_state_dict(result['optimizer_state_dict'])
        #     scheduler.load_state_dict(result['scheduler_state_dict'])
        #     logger = result['logger']
        # else:
        #     logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
        else:
            server = make_server(model)
            client,supervised_clients = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        client,supervised_clients = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    cfg['global']['num_epochs'] = cfg['cycles']  
    mode = cfg['loss_mode']

    #train supervised clients 
    for epoch in range(last_epoch, cfg['switch_epoch_pred'] + 1):
        train_client(batchnorm_dataset, client_dataset['train'], server, client, supervised_clients, optimizer, metric, logger, epoch,mode)
        # if 'ft' in cfg and cfg['ft'] == 0:
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        #     logger.reset()
        #     server.update_parallel(client)
        # else:
            # logger.reset()
            # server.update(client)
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        logger.reset()
        server.update(client)
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(data_loader['test'], test_model, metric, logger, epoch)
        # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'scheduler_state_dict': scheduler.state_dict(),
        #           'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
        # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'scheduler_state_dict': scheduler.state_dict(),
        #           'data_split': data_split, 'logger': logger}
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'data_split': data_split, 'logger': logger,'supervised_clients':supervised_clients }
        # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
        #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
        #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
        #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
        if epoch==50 or epoch==51:
            print('saving')
            save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                            './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    # return
    # torch.cuda.empty_cache()
    #Get Class impressions
    cfg['loss_mode'] = 'gen'
    net = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    net.load_state_dict(server.model_state_dict)
    # net.load_state_dict(result['model_state_dict'])
    net.to(cfg['device'])
    net.eval()

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    args = dict()
    args['verifier'] = False
    args['adi_scale'] = 0
    # if args.verifier and args.adi_scale == 0:
    #     # if multiple GPUs are used then we can change code to load different verifiers to different GPUs
    #     if args.local_rank == 0:
    #         print("loading verifier: ", args.verifier_arch)
    #         net_verifier = models.__dict__[args.verifier_arch](pretrained=True).to(cfg['device'])
    #         net_verifier.eval()

    #         if use_fp16:
    #             net_verifier = net_verifier.half()

    # if args.adi_scale != 0.0:
    #     student_arch = "resnet18"
    #     net_verifier = models.__dict__[student_arch](pretrained=True).to(device)
    #     net_verifier.eval()

    #     if use_fp16:
    #         net_verifier, _ = amp.initialize(net_verifier, [], opt_level="O2")

    #     net_verifier = net_verifier.to(device)
    #     net_verifier.train()

    #     if use_fp16:
    #         for module in net_verifier.modules():
    #             if isinstance(module, nn.BatchNorm2d):
    #                 module.eval().half()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    from deepinversion import DeepInversionClass
    from cifar10inversion import get_images
    # exp_name = args.exp_name
    exp_name = 'rn9_2001_samples_take0.1_l20.0001'
    # final images will be stored here:
    adi_data_path = "./final_images/%s"%exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s"%exp_name

    args['iterations'] = 2000
    args['start_noise'] = True
    # args.detach_student = False

    args['resolution'] = 32
    # bs = args.bs
    bs = 500
    jitter = 30
    parameters = dict()
    parameters["resolution"] = 32
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = False
    # parameters["do_flip"] = args.do_flip
    # parameters["random_label"] = args.random_label
    # parameters["random_label"] = True
    parameters["store_best_images"] = True
    # parameters["store_best_images"] = args.store_best_images

    criterion = torch.nn.CrossEntropyLoss()

    # coefficients = dict()
    # coefficients["r_feature"] = args.r_feature
    # coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    # coefficients["tv_l1"] = args.tv_l1
    # coefficients["tv_l2"] = args.tv_l2
    # coefficients["l2"] = args.l2
    # coefficients["lr"] = args.lr
    # coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    # coefficients["adi_scale"] = args.adi_scale
    coefficients = dict()
    coefficients["r_feature"] = 0.01    
    coefficients["first_bn_multiplier"] = 10
    coefficients["tv_l1"] =0.0
    coefficients["tv_l2"] =0.001
    coefficients["l2"] = 0.00001
    coefficients["lr"] = 0.1
    coefficients["main_loss_multiplier"] = 1
    coefficients["adi_scale"] = 0.0
    network_output_function = lambda x: x

    # check accuracy of verifier
    # if args.verifier:
    #     hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    # else:
    #     hook_for_display = None
    hook_for_display = None
    ###################################################################
    criterion = torch.nn.CrossEntropyLoss()
    discp = 'cifar10__500testlr_images_3000_lr0.05_vs_25-6_l23e-8'
    prefix = "runs/data_generation/"+discp+"/"

    for create_folder in [prefix, prefix+"/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)
    # place holder for inputs
    data_type = torch.float
    inputs = torch.randn((bs, 3, 32, 32), requires_grad=True, device=cfg['device'], dtype=data_type)
    net_student=None
    train_writer = None  # tensorboard writter
    global_iteration = 0
    optimizer = torch.optim.Adam([inputs], lr=0.05)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = None
    print("Starting model inversion")
    # torch.cuda.empty_cache()
    for i in range(1):
        class_impressions_,targets_ = get_images(net=net, bs=bs, epochs=args['iterations'], idx=0,
                            net_student=net_student, prefix=prefix, competitive_scale=0.0,
                            train_writer=train_writer, global_iteration=global_iteration, use_amp= False,
                            optimizer=optimizer,scheduler=scheduler, inputs=inputs, bn_reg_scale=10,
                            var_scale=25e-6, random_labels=False, l2_coeff=3e-8,store_best_images=False)
        if i==0:
            class_impressions,targets = class_impressions_,targets_
        else:
            print(i)
            class_impressions=torch.cat([class_impressions,class_impressions_],dim=0)
            targets=torch.cat([targets,targets_],dim=0)
        print(class_impressions.shape)
    #####################################################################
    # DeepInversionEngine = DeepInversionClass(net_teacher=net,
    #                                          final_data_path=adi_data_path,
    #                                          path=exp_name,
    #                                          parameters=parameters,
    #                                          setting_id=0,
    #                                          bs = bs,
    #                                          use_fp16 = False,
    #                                          jitter = jitter,
    #                                          criterion=criterion,
    #                                          coefficients = coefficients,
    #                                          network_output_function = network_output_function,
    #                                          hook_for_display = hook_for_display)
    # net_student=None
    # # if args.adi_scale != 0:
    # #     net_student = net_verifier
    # targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] *int(bs//10)).to(cfg['device'])
    # print(targets)
    # class_impressions = DeepInversionEngine.generate_batch(net_student=net_student,targets = targets)

    #############################################################################################################

    # CI_dataset =  torch.utils.data.TensorDataset(class_impressions.detach().cpu(), targets.detach().cpu())
    # CI_dataset.data = class_impressions.detach().cpu().numpy()
    # CI_dataset.target = targets.detach().cpu().numpy()
    # CI_dataset.other['id'] = list(range(len(CI_dataset.data)))
    # transform = FixTransform(cfg['data_name'])
    # CI_dataset.transform = transform
    ###############################################################################################################
    root = './data/{}'.format(cfg['data_name'])
    # class_impressions =  numpy.transpose(class_impressions, (0,2, 3, 1))
    CI_dataset = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(cfg['data_name']))
    # CI_dataset.data = class_impressions.detach().cpu().numpy()
    CI_dataset.data = class_impressions.permute(0,2,3,1).to("cpu", torch.uint8).numpy()
    print(CI_dataset.data.shape)
    CI_dataset.target = targets.detach().cpu().numpy()
    CI_dataset.other['id'] = list(range(len(CI_dataset.data)))
    transform = FixTransform(cfg['data_name'])
    CI_dataset.transform = transform
    print(CI_dataset)
    cfg['resume_mode'] = 1
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'],load_tag='best')
        # model_tag = '0_CIFAR10_resnet9_50000_sup_100_0.1_iid_5-5_0.2_1'
        # # load_tag = 'checkpoint'
        # load_tag = 'best'
        # path ='/home/sampathkoti/codes/SemiFL/output/model/'
        # if os.path.exists('{}/{}_{}.pt'.format(path,model_tag, load_tag)):
        #     result = load('{}/{}_{}.pt'.format(path,model_tag, load_tag))
        # else:
        #     print('Not exists model tag: {}, start from scratch'.format(model_tag))
        #     from datetime import datetime
        #     from logger import Logger
        #     last_epoch = 1
        #     logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        #     logger = Logger(logger_path)
        #     result = {'epoch': last_epoch, 'logger': logger}
        # if True:
        #     print('Resume from {}'.format(result['epoch']))
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            # supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            supervised_clients = result['supervised_clients']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        # if last_epoch > 1:
        #     model.load_state_dict(result['model_state_dict'])
        #     optimizer.load_state_dict(result['optimizer_state_dict'])
        #     scheduler.load_state_dict(result['scheduler_state_dict'])
        #     logger = result['logger']
        # else:
        #     logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
        else:
            server = make_server(model)
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        client,supervised_clients = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    cfg['global']['num_epochs'] = cfg['cycles']  
    cfg['loss_mode'] = 'alt-fix'
    mode  = 'alt-fix'
    # cfg['loss_mode'] = 'fix-mix'
    # mode = 'fix-mix'
    for epoch in range(cfg['switch_epoch_pred']+1, cfg['global']['num_epochs'] + 1):
        # if mode == 'sim-ft-fix' or mode == 'sup-ft-fix':
        #     # print('entered fix-mix',epoch)
        #     if epoch<=cfg['switch_epoch_pred']:
        #         if 'sim' in mode:
        #             cfg['loss_mode'] = 'sim-ft'
        #         elif 'sup' in mode:
        #             cfg['loss_mode'] = 'sup-ft'
        #     elif epoch > cfg['switch_epoch_pred']:
        #         # print('entered fix-mix',epoch)
        #         cfg['loss_mode'] = 'alt-fix'
        train_client(batchnorm_dataset, client_dataset['train'], server, client, supervised_clients, optimizer, metric, logger, epoch,mode,CI_dataset)
        # train_client(batchnorm_dataset, client_dataset['train'], server, client, supervised_clients, optimizer, metric, logger, epoch,mode)
        # if 'ft' in cfg and cfg['ft'] == 0:
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        #     logger.reset()
        #     server.update_parallel(client)
        # else:
            # logger.reset()
            # server.update(client)
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        logger.reset()
        server.update(client)
        # cfg['loss_mode'] = 'train-server'
        # train_server(CI_dataset, server, optimizer, metric, logger, epoch)
        # cfg['loss_mode'] = 'fix-mix'
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(data_loader['test'], test_model, metric, logger, epoch)
        # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'scheduler_state_dict': scheduler.state_dict(),
        #           'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
        # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'scheduler_state_dict': scheduler.state_dict(),
        #           'data_split': data_split, 'logger': logger}
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'data_split': data_split, 'logger': logger,'supervised_clients':supervised_clients }
        # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
        #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
        #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
        #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
        if epoch%10==0:
            print('saving')
            save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                            './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def make_server(model):
    server = Server(model)
    return server


def make_client(model, data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]})
    num_supervised_clients = int(cfg['num_supervised_clients'])
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_supervised_clients]].tolist()
    print(client_id)
    for i in range(num_supervised_clients):
        client[client_id[i]].supervised = True
        
    return client , client_id


def train_client(batchnorm_dataset, client_dataset, server, client, supervised_clients, optimizer, metric, logger, epoch,mode,CI_dataset=None):
    logger.safe(True)
    if 'ft' in cfg['loss_mode']:
        if epoch <= cfg['switch_epoch']:
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
        else:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
    elif 'at' in cfg['loss_mode']:
        cfg['srange'] = [21,31,51,61,81,91,111,121]
        if cfg['srange'][0]<=epoch<=cfg['srange'][1] or cfg['srange'][2]<=epoch<=cfg['srange'][3] or cfg['srange'][4]<=epoch<=cfg['srange'][5] or cfg['srange'][6]<=epoch<=cfg['srange'][7]:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
        else:
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
    elif 'fix' in cfg['loss_mode'] and 'alt' not in mode:
        if epoch >cfg['switch_epoch_pred']:
            cfg['loss_mode'] = 'fix-mix'
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = ACL-set(supervised_clients)
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
        else:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
    elif 'alt-fix' in mode:
        print('entered alt-fix mode')
        if epoch %2 == 0:
            cfg['loss_mode'] = 'fix-mix'
            print(cfg['loss_mode'])
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = ACL-set(supervised_clients)
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
        elif epoch % 2 != 0:
            cfg['loss_mode'] = 'sup'
            print(cfg['loss_mode'])
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True

    # server.distribute(client, batchnorm_dataset)
    print(f'traning the following clients {client_id}')
    server.distribute(client,batchnorm_dataset)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        # print(type(client[m].data_split['train']))
        dataset_m = separate_dataset(client_dataset, client[m].data_split['train'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            # cfg['pred'] = True
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
            # cfg['pred'] = False
        # print(cfg)
        # print(dataset_m is not None)
        if dataset_m is not None:
            # print(cfg)
            # print(dataset_m)
            if cfg['loss_mode'] == 'fix-mix' and dataset_m[0] is not None and dataset_m[1] is not None:
                client[m].active = True
                client[m].trainntune(dataset_m, lr, metric, logger, epoch,CI_dataset)
            elif 'sim' in cfg['loss_mode'] or 'sup' in cfg['loss_mode']:
                client[m].active = True
                client[m].trainntune(dataset_m, lr, metric, logger, epoch)
            else:
                client[m].active = False

        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def train_server(dataset, server, optimizer, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    server.train(dataset, lr, metric, logger)
    _time = (time.time() - start_time)
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            input['loss_mode'] = cfg['loss_mode']
            input['supervised_mode'] = False
            input['test'] = True
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
