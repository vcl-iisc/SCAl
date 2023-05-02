import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, separate_dataset_su, make_batchnorm_stats
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    # print(len(dataset['test']))
    process_dataset(dataset)
    dataset['train'], _, supervised_idx = separate_dataset_su(dataset['train'])
    # print(len(supervised_idx))
    data_loader = make_data_loader(dataset, 'global')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    net = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    # print(model)
    # print(cfg['local'].keys())
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'],'best')
        last_epoch = result['epoch']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    cfg['model_name'] = 'global'
    # print(list(model.buffers()))
    cfg['global']['num_epochs'] = cfg['cycles']  
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        # cfg['model_name'] = 'local'
        logger.safe(True)
        train(data_loader['train'], model, optimizer, metric, logger, epoch)
        # module = model.layer1[0].n1
        # print(list(module.named_buffers()))
        # print(list(model.buffers()))
        test_model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
        # print(list(model.buffers()))
        # module = model.layer1[0].n1
        # print(list(module.named_buffers()))
        test(data_loader['test'], test_model, metric, logger, epoch)
        # print(list(model.buffers()))
        # module = model.layer1[0].n1
        # print(list(module.named_buffers()))
        scheduler.step()
        logger.safe(False)
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'supervised_idx': supervised_idx,
                  'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(), 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    #Get Class impressions
    cfg['loss_mode'] = 'gen'
    
    net.load_state_dict(result['model_state_dict'])
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

    from deepinversion import DeepInversionClass
    from cifar10inversion import get_images
    # exp_name = args.exp_name
    exp_name = 'rn9_test3_inversion'
    # final images will be stored here:
    adi_data_path = "./final_images/%s"%exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s"%exp_name

    args['iterations'] = 200000
    args['start_noise'] = True
    # args.detach_student = False

    args['resolution'] = 32
    # bs = args.bs
    bs = 256
    jitter = 30
    parameters = dict()
    parameters["resolution"] = 32
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = False
    # parameters["do_flip"] = args.do_flip
    # parameters["random_label"] = args.random_label
    parameters["random_label"] = True
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
    coefficients["tv_l2"] =0.0001
    coefficients["l2"] = 0.00001  
    coefficients["lr"] = 0.25
    coefficients["main_loss_multiplier"] = 1
    coefficients["adi_scale"] = 0.0
    network_output_function = lambda x: x

    # check accuracy of verifier
    # if args.verifier:
    #     hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    # else:
    #     hook_for_display = Nonee?
    hook_for_display = None
    ###################################################################
    criterion = torch.nn.CrossEntropyLoss()
    discp = 'cifar10_50k'
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
    print("Starting model inversion")
    inputs = get_images(net=net, bs=bs, epochs=args['iterations'], idx=0,
                        net_student=net_student, prefix=prefix, competitive_scale=0.0,
                        train_writer=train_writer, global_iteration=global_iteration, use_amp= False,
                        optimizer=torch.optim.Adam([inputs], lr=0.1), inputs=inputs, bn_reg_scale=10,
                        var_scale=25e-6, random_labels=False, l2_coeff=0.0)
    return


def train(data_loader, model, optimizer, metric, logger, epoch):
    model.train(True)
    start_time = time.time()
    model.projection.requires_grad_(False)
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        input['loss_mode'] = cfg['loss_mode']
        output = model(input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return


def test(data_loader, model, metric, logger, epoch):
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
            input['test'] = True
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
