import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from .early_stopping import EarlyStopping

sys.path.append("../../AnomalyLLM")
def MixTrainer(model, model_optimizer, train_dl, val_dl_list, test_dl_list, device, logger, config, idx):
    # Start training
    logger.debug("Training started ....")

    save_path = "./best_network/" + config.dataset
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    all_epoch_train_loss, all_epoch_test_loss = [], []
    # center = torch.zeros(config.project_channels, device=device)
    # center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss, length = model_train(model, model_optimizer, train_dl,
                                                                    length, config, device, epoch)
        all_epoch_train_loss.append(train_loss.item())
        scheduler.step(train_loss)
        logger.debug(f'\nEpoch : {epoch}\n'
                f'Train Loss     : {train_loss:.4f}\t | \n'
                )
        
        total_accumulator = reasonable_accumulator()
        test_origin_list, test_affiliation_list, test_score_list, pred_list, score_reasonable_list = [], [], [], [], []

        for  i in range(len(test_dl_list)):
            test_dl = test_dl_list[i]
            # val_target, val_score_origin, val_loss, all_projection = model_evaluate(model, val_dl,  length, config,
            #                                                                     device, epoch)
            test_target, test_score_origin, test_loss, all_projection = model_evaluate(model, test_dl, length,
                                                                                    config, device, epoch)
            test_origin_list.append(test_score_origin)
        # if epoch < config.change_center_epoch:
        #     center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
            logger.debug(f'Scene Id: {i+1} \n')
            logger.debug(f'Test Loss     : {test_loss:.4f}\t  | \n')
        
        # all_epoch_test_loss.append(val_loss.item())
            if config.dataset == 'UCR':
                # val_affiliation, val_score, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                #                                            config.detect_nu)
                test_affiliation, test_score, predict = ad_predict(test_target, test_score_origin,
                                                                config.threshold_determine, config.detect_nu)
                score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
                # score_reasonable_list.append(score_reasonable)
                total_accumulator.cnt += score_reasonable.cnt
                total_accumulator.correct_num += score_reasonable.correct_num
                test_affiliation_list.append(test_affiliation)
                test_score_list.append(test_score)
                pred_list.append(predict)

                test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
                test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
                test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)
                print("Test accuracy metrics")
                
                # early_stopping(score_reasonable, test_affiliation, test_score, model)
                print("Test affiliation-metrics")
                logger.debug(
                    f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
                print("Test RAP F1")
                logger.debug(
                    f'Test F1: {test_f1:2.4f}  | \tTest precision: {test_precision:2.4f}  | \tTest recall: {test_recall:2.4f}\n')
        
        logger.debug(f'Test accuracy: {total_accumulator.correct_num/total_accumulator.cnt:2.4f}\n')
        early_stopping(total_accumulator, test_affiliation_list, test_score_list, model)
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
        # val_affiliation, val_score, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
        #                                            config.detect_nu)
        test_affiliation, test_score, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                           config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
    # val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
    # val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
    # val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
    # print("Valid affiliation-metrics")
    # logger.debug(
    #     f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
    # print("Valid RAP F1")
    # logger.debug(f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    test_f1 = sum([score.f1(ScoreType.RevisedPointAdjusted) for score in test_score])/len(test_score)
    test_precision = sum([score.precision(ScoreType.RevisedPointAdjusted) for score in test_score])/len(test_score)
    test_recall = sum([score.recall(ScoreType.RevisedPointAdjusted) for score in test_score])/len(test_score)
    aff_precision = sum([aff["precision"] for aff in test_affiliation])/len(test_affiliation)
    aff_recall = sum([aff["recall"] for aff in test_affiliation])/len(test_affiliation)
    print("Test affiliation-metrics")
    logger.debug(
        f'Test precision: {aff_precision:2.4f}  | \tTest recall: {aff_recall:2.4f}\n')
    print("Test RAP F1")
    logger.debug(f'Test F1: {test_f1:2.4f}  | \tTest precision: {test_precision:2.4f}  | \tTest recall: {test_recall:2.4f}\n')

    # writer = SummaryWriter()
    # for i in range(config.num_epoch):
    #     writer.add_scalars('loss', {'train': all_epoch_train_loss[i],
    #                                 'test': all_epoch_test_loss[i]}, i)
    # # writer.add_embedding(part_embedding_feature, metadata=part_embedding_target, tag='test embedding')
    # writer.close()

    return test_origin_list, test_affiliation, test_score, score_reasonable, pred_list


def model_train(model, model_optimizer, train_loader, length, config, device, epoch):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device) # (B, T)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        all_data = torch.cat((data, aug1, aug2), dim=0) # （B*3, T, D）
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


def train(feature1, center, length, epoch, config, device):
    # normalize feature vectors
    center = F.normalize(center, dim=1)
    feature1 = F.normalize(feature1, dim=1)
    num_data = len(center) // 3

    distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
    distance1[:num_data] = 1 - distance1[:num_data]

    # Prevent model collapse
    # sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
    # sigma_aug2 = torch.sqrt(feature_dec1.var([0]) + 0.0001)
    # sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
    # sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
    # loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)

    center_n = center[:num_data]
    center_1 = center[num_data:num_data*2]
    center_2 = center[num_data*2:]
    center_distance = 2 - F.cosine_similarity(center_n, center_1, eps=1e-6) - F.cosine_similarity(center_n, center_2, eps=1e-6)
    loss_sigam = torch.mean(center_distance)

    # The Loss function that representations reconstruction
    score = distance1 
    if config.objective == 'soft-boundary':
        diff1 = score - length
        loss_oc = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
    else:
        loss_oc = torch.mean(score)
    loss = config.omega1 * loss_oc + config.omega2 * loss_sigam
    return loss, score

def test(feature1, center, length, epoch, config, device):
    # normalize feature vectors
    center = F.normalize(center, dim=1)
    feature1 = F.normalize(feature1, dim=1)
    

    distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
    distance1 = 1 - distance1

    # Prevent model collapse
    # sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
    # sigma_aug2 = torch.sqrt(feature_dec1.var([0]) + 0.0001)
    # sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
    # sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
    # loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)
    loss_sigam = 0

    # The Loss function that representations reconstruction
    score = distance1 
    if config.objective == 'soft-boundary':
        diff1 = score - length
        loss_oc = length + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
    else:
        loss_oc = torch.mean(score)
    loss = config.omega1 * loss_oc + config.omega2 * loss_sigam
    return loss, score




def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)



