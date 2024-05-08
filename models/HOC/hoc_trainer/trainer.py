import sys
import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from .early_stopping import EarlyStopping
from kmeans_pytorch import kmeans
import time

sys.path.append("../../AnomalyLLM")
def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, logger, config, idx):
    # Start training
    logger.debug("Training started ....")

    save_path = "./best_network/" + config.dataset
    if config.dataset == 'UCR':
        save_path = os.path.join(save_path, config.subdataset) 
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    all_epoch_train_loss, all_epoch_test_loss = [], []
    # center = torch.zeros(config.project_channels, device=device)
    # center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.


    if config.reclass == 'memtrans' and config.mem_init == True:
        model.representer.mem.data.copy_(memory_initialize(train_dl, config.num_memory, device))

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss, length = model_train(model, model_optimizer, train_dl,
                                                                    length, config, device, epoch)
        val_target, val_score_origin, val_loss, all_projection = model_evaluate(model, val_dl,  length, config,
                                                                            device, epoch)
        test_target, test_score_origin, test_loss, all_projection = model_evaluate(model, test_dl, length,
                                                                                   config, device, epoch)

        # if epoch < config.change_center_epoch:
        #     center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
        scheduler.step(train_loss)
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \n'
                     f'Valid Loss     : {val_loss:.4f}\t  | \n'
                     f'Test Loss     : {test_loss:.4f}\t  | \n'
                     )
        all_epoch_train_loss.append(train_loss.item())
        all_epoch_test_loss.append(val_loss.item())
        if config.dataset == 'UCR':
            val_affiliation, val_score, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            test_affiliation, test_score, predict = ad_predict(test_target, test_score_origin,
                                                               config.threshold_determine, config.detect_nu)
            score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
            test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
            test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
            test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)
            print("Test accuracy metrics")
            logger.debug(
                f'Test accuracy: {score_reasonable.correct_num:2.4f}\n')
            early_stopping(score_reasonable, test_affiliation, test_score, model)
            print("Test affiliation-metrics")
            logger.debug(
                f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
            print("Test RAP F1")
            logger.debug(
                f'Test F1: {test_f1:2.4f}  | \tTest precision: {test_precision:2.4f}  | \tTest recall: {test_recall:2.4f}\n')
            if early_stopping.early_stop:
                print("Early stopping")
                break

    logger.debug("\n################## Training is Done! #########################")
    # according to scores to create predicting labels
    if config.dataset == 'UCR':
        score_reasonable = early_stopping.best_score
        test_affiliation = early_stopping.best_affiliation
        test_score = early_stopping.best_rpa_score
        print("Test accuracy metrics")
        logger.debug(
            f'Test accuracy: {score_reasonable.correct_num:2.4f}\n')
    else:
        val_affiliation, val_score, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                   config.detect_nu)
        test_affiliation, test_score, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                           config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
    val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
    val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
    val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
    print("Valid affiliation-metrics")
    logger.debug(
        f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
    print("Valid RAP F1")
    logger.debug(f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')

    test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
    test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
    test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test affiliation-metrics")
    logger.debug(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    print("Test RAP F1")
    logger.debug(f'Test F1: {test_f1:2.4f}  | \tTest precision: {test_precision:2.4f}  | \tTest recall: {test_recall:2.4f}\n')

    # writer = SummaryWriter()
    # for i in range(config.num_epoch):
    #     writer.add_scalars('loss', {'train': all_epoch_train_loss[i],
    #                                 'test': all_epoch_test_loss[i]}, i)
    # # writer.add_embedding(part_embedding_feature, metadata=part_embedding_target, tag='test embedding')
    # writer.close()

    return test_score_origin, test_affiliation, test_score, score_reasonable, predict

def memory_initialize(dl, n_memory, device):
    all_data = []
    for batch_idx, (data, target, aug1, aug2) in enumerate(dl):
        # send to device
        data = data.float().to(device) # (B, d, T)
        all_data.append(data)
    all_data = torch.cat(all_data,dim=0) # (N, d, T)
    T = all_data.size(2)

    memory_init = []
    for i in range(all_data.size(1)):
        memory_init.append(k_means_clustering(x=all_data[:,i], n_mem=n_memory, d_model=T).unsqueeze(0))
    return torch.cat(memory_init,dim=0)

def k_means_clustering(x,n_mem,d_model):
    start = time.time()

    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    # sckit-learn xxxx (cuda problem)
    _, cluster_centers = kmeans(X=x, num_clusters=n_mem, distance='euclidean', device=torch.device('cuda:0'))
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers


def model_train(model, model_optimizer, train_loader, length, config, device, epoch):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device) # (B, T)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        all_data = torch.cat((data, aug1, aug2), dim=0) # （B*3, d, T）
        # optimizer
        model_optimizer.zero_grad()
        feature1, center = model(all_data)
        loss, score = train(feature1, center, length, epoch, config, device)
        # Update hypersphere radius R on mini-batch distances
        if (config.objective == 'soft-boundary') and (epoch >= config.freeze_length_epoch):
            length = torch.tensor(get_radius(score, config.nu), device=device)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

        target = target.reshape(-1)

        predict = score.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        all_target.extend(target)
        all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()

    return all_target, all_predict, total_loss, length


def model_evaluate(model, test_dl, length, config, device, epoch):
    model.eval()
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    all_projection = []
    with torch.no_grad():
        for data, target, aug1, aug2 in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            feature1, center = model(data)
            loss, score = test(feature1, center, length, epoch, config, device)
            total_loss.append(loss.item())
            predict = score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(predict)
            all_projection.append(feature1)

    total_loss = torch.tensor(total_loss).mean()  # average loss
    all_projection = torch.cat(all_projection, dim=0)
    all_target = np.array(all_target)

    return all_target, all_predict, total_loss, all_projection


def train(feature, center, length, epoch, config, device):
    # normalize feature vectors
    num_data = len(center) // 3
    '''Knowledge distillation loss'''
    # center = F.normalize(center, dim=1)
    # feature = F.normalize(feature, dim=1)
    # distance1 = F.cosine_similarity(feature, center, eps=1e-6)
    # # distance1 = 1 - distance1
    # distance1[:num_data] = 1 - distance1[:num_data]
    dist1 = torch.norm(center-feature, p=2, dim=1)
    dist2 = 1 - torch.exp(-dist1)
    distance1 = torch.cat((dist1[:num_data], -torch.log(dist2[num_data:] + 1e-6)),dim=0)


    # Prevent model collapse
    # sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
    # sigma_aug2 = torch.sqrt(feature_dec1.var([0]) + 0.0001)
    # sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
    # sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
    # loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)
    
    '''Negative-sample-free contrastive loss'''
    center = F.normalize(center, dim=1)
    center_n = center[:num_data]
    center_1 = center[num_data:num_data*2]
    center_2 = center[num_data*2:]
    center_distance = 2 - F.cosine_similarity(center_n, center_1, eps=1e-6) - F.cosine_similarity(center_n, center_2, eps=1e-6)
    loss_ctr = torch.mean(center_distance)
    # dis_loss = nn.MSELoss()
    # loss_ctr = dis_loss(center_n, center_1) + dis_loss(center_n, center_2)

    '''Feature loss'''
    # feature_n = feature[:num_data]
    # feature_1 = feature[num_data:num_data*2]
    # feature_2 = feature[num_data*2:]
    # feature_distance = F.cosine_similarity(feature_n, feature_1, eps=1e-6) + F.cosine_similarity(feature_n, feature_2, eps=1e-6)
    # loss_feature = torch.mean(feature_distance)
    loss_feature = 0

    # The Loss function that representations reconstruction
    score = distance1 
    if config.objective == 'soft-boundary':
        diff1 = score - length
        loss_kd = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
    else:
        loss_kd = torch.mean(score)
    loss = config.omega1 * loss_kd + config.omega2 * loss_ctr + config.omega3 * loss_feature
    return loss, score

def test(feature1, center, length, epoch, config, device):
    # normalize feature vectors
    '''Knowledge distillation loss'''
    # center = F.normalize(center, dim=1)
    # feature1 = F.normalize(feature1, dim=1)
    # distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
    # distance1 = 1 - distance1
    distance1 = torch.norm(center-feature1, p=2, dim=1)

    # The Loss function that representations reconstruction
    score = distance1 
    if config.objective == 'soft-boundary':
        diff1 = score - length
        loss_kd = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
    else:
        loss_kd = torch.mean(score)
    loss = loss_kd
    return loss, score


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)



