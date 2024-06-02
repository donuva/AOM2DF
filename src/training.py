from datetime import datetime
import numpy as np
from torch.cuda.amp import autocast
import src.model.utils as utils
import src.eval_utils as eval_utils
import src.eval_utils as eval_utils
import torch
import os
from src.utils import save_training_data


def fine_tune(epoch,
              model,
              img_encoder,
              train_loader,
              train_gt_span,
              metric,
              optimizer,
              device,
              args,
              logger=None,
              callback=None,
              log_interval=1,
              tb_writer=None,
              tb_interval=1,
              scaler=None):

    total_step = len(train_loader)
    model.train()
    img_encoder.train()
    img_encoder.zero_grad()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        input_ids, attention_masks, image_feats, span_labels, span_masks,data_ids,sentiment_value,noun_mask,dependency_matrix = batch
        batch_gt_spans = []
        for j_id in data_ids:
            batch_gt_spans.append(train_gt_span[j_id])
        mner_encode = {}
        mner_encode['labels'] = span_labels
        mner_encode['masks'] = span_masks
        mner_encode['spans'] = batch_gt_spans

        with torch.no_grad():
            imgs_f=[x.numpy().tolist() for x in image_features]
            imgs_f=torch.tensor(imgs_f).to(device)
            imgs_f, img_mean, img_att = img_encoder(imgs_f)
            img_att=img_att.view(-1, 2048, 49).permute(0, 2, 1)

        with autocast(enabled=args.amp):
            loss = model.forward(
                input_ids=input_ids.to(device),
                image_features=list(map(lambda x: x.to(device), img_att)),
                sentiment_value=sentiment_value.to(device) if sentiment_value is not None else None,
                noun_mask=noun_mask.to(device),
                attention_mask=attention_mask.to(device),
                dependency_matrix=dependency_matrix.to(device),
                aesc_infos=mner_encode)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, args.epochs, i + 1, total_step, loss.item()))
        # Backward and optimize
        cur_step = i + 1 + epoch * total_step
        t_step = args.epochs * total_step
        liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
        utils.set_lr(optimizer, liner_warm_rate * args.lr)

        optimizer.zero_grad()

        loss.backward()
        utils.clip_gradient(optimizer, args.grad_clip)

        optimizer.step()


def trc_pretrain(epochs,
             model,
             img_encoder,
             train_loader,
             optimizer,
             device,
             args,
             logger=None,
             callback=None,
             log_interval=1,
             tb_writer=None,
             tb_interval=1,
             scaler=None):
    start=datetime.now()
    total_step = len(train_loader)*epochs
    model.train()
    img_encoder.train()
    img_encoder.zero_grad()
    min_loss = 100
    epoch=0
    global_step = 0
    criterion=torch.nn.CrossEntropyLoss(reduction='mean')
    start_time = datetime.now()
    while epoch<epochs:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        for i, batch in enumerate(train_loader):
            # Forward pass
            global_step+=1
            with torch.no_grad():
                imgs_f=[x.numpy().tolist() for x in batch['image_features']]
                imgs_f=torch.tensor(imgs_f).to(device)
                imgs_f, img_mean, img_att = img_encoder(imgs_f)
                img_att=img_att.view(-1, 2048, 49).permute(0, 2, 1)
            with autocast(enabled=args.amp):
                logits = model.forward(
                    input_ids=batch['input_ids'].to(device),
                    image_features=list(map(lambda x: x.to(device), img_att)),
                    noun_mask=batch['noun_mask'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
                loss=criterion(logits.view(-1,2),torch.tensor(batch['ifpairs']).to(args.device))
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()


        logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs,loss.item()))
        if loss.item() < min_loss:
            min_loss=loss.item()
            current_checkpoint_path = os.path.join(
                args.checkpoint_path, ('model{}_minloss').format(epoch))
            model.save_pretrained(current_checkpoint_path)
            save_training_data(path=current_checkpoint_path,
                               optimizer=optimizer,
                               scaler=scaler,
                               epoch=epoch)
            logger.info('Saved checkpoint at "{}"'.format(args.checkpoint_path))
        # save checkpoint
        elif epoch % args.checkpoint_every == 0:
            if args.bart_init == 0:
                current_checkpoint_path = os.path.join(
                    args.checkpoint_path, 'model{}random_again'.format(epoch))
            else:
                current_checkpoint_path = os.path.join(
                    args.checkpoint_path, ('model{}').format(epoch))
            if args.cpu:
                model.save_pretrained(current_checkpoint_path)
            else:
                model.save_pretrained(current_checkpoint_path)
            save_training_data(path=current_checkpoint_path,
                               optimizer=optimizer,
                               scaler=scaler,
                               epoch=epoch)
            logger.info('Saved checkpoint at "{}"'.format(args.checkpoint_path))
        epoch += 1
    logger.info("Finish pretraining  " + str(datetime.now() - start), pad=True)



def save_finetune_model(model):
    torch.save(model.state_dict(),'/home/zhouru/ABSA3/save_model/best_model.pth')


def save_img_encoder(args,img_encoder):
    file_name=os.path.join(args.checkpoint_path,'resnet152.pth')
    torch.save(img_encoder.state_dict(),file_name)
    pass