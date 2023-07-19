import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from torchvision.ops.boxes import box_area
import torch.nn.functional as F
from torch import nn

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, metrics, is_best, final_args):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param model: model
    :param optimizer: optimizer to update model's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'metrics': metrics,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)

def save_clf_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, Acc, final_args):
    """
    Saves model checkpoint.
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'Acc': Acc,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def calc_acc(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc


def calc_classwise_acc(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    classwise_acc = matrix.diagonal()/matrix.sum(axis=1)
    return classwise_acc


def calc_map(y_true, y_scores):
    mAP = average_precision_score(y_true, y_scores,average=None)
    return mAP

def calc_precision_recall_fscore(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division = 1)
    return(precision, recall, fscore)

def giou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()

def mIoU_xyxy(box_a, box_b):
    # inter = intersection(box_a, box_b)
    assert box_a.shape == box_b.shape
    (m, n) = box_a.shape
    iou_sum = 0
    for i in range(m):
        x1 = max(box_a[i, 0], box_b[i, 0])
        y1 = max(box_a[i, 1], box_b[i, 1])
        x2 = min(box_a[i, 2], box_b[i, 2])
        y2 = min(box_a[i, 3], box_b[i, 3])
        if x1 >= x2 or y1 >= y2:
            inter = 0.0
        inter = float((x2 - x1 + 1) * (y2 - y1 + 1))
        box_a_area = (box_a[i, 2] - box_a[i, 0] + 1) * (box_a[i, 3] - box_a[i, 1] + 1)
        box_b_area = (box_b[i, 2] - box_b[i, 0] + 1) * (box_b[i, 3] - box_b[i, 1] + 1)
        union = box_a_area + box_b_area - inter
        iou = inter / float(max(union, 1))
        iou_sum = iou_sum + iou

    m_iou = iou_sum / m
    return m_iou

def mIoU_single(box_a, box_b):
    # inter = intersection(box_a, box_b)
    assert box_a.shape == box_b.shape
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x1 >= x2 or y1 >= y2:
        inter = 0.0
    inter = float((x2 - x1 + 1) * (y2 - y1 + 1))
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = box_a_area + box_b_area - inter
    iou = inter / float(max(union, 1))
    return iou

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def loss_giou_l1(outputs, targets):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """

    src_boxes = outputs
    target_boxes = targets
    (num_boxes, n) = src_boxes.shape
    loss_l1= F.l1_loss(src_boxes, target_boxes, reduction='none')
    losses_l1 = loss_l1.sum() / num_boxes
    loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
    losses_giou = loss_giou.sum() / num_boxes
    losses = losses_l1 + losses_giou
    
    return losses

def compute_acc(gt, pred):
    assert len(gt) == len(pred)
    num = len(gt)
    true_num = 0
    for i in range(num):
        if gt[i] == pred[i]:
            true_num = true_num + 1
    acc_score = true_num / num
    return acc_score

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return {"bbox": x}


# def evaluation(test_acc, epoch, test_fscore, test_bbox_miou):

#     test_avg_acc_miou = 0.5 * (test_acc + test_bbox_miou)
#     best_accuracy = 0
#     best_accuracy_epoch = 0
#     best_miou = 0
#     best_miou_epoch = 0
#     best_avg_acc_miou = 0
#     best_avg_acc_miou_epoch = 0
    
#     if test_acc >= best_accuracy:  
#         best_accuracy = test_acc
#         best_accuracy_epoch = epoch
#         best_acc_epoch_fscore = test_fscore
#         best_acc_epoch_miou = test_bbox_miou
#         best_acc_epoch_avg_acc_miou = test_avg_acc_miou
    
#     print('Best Acc epoch:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_accuracy_epoch, best_accuracy, best_acc_epoch_fscore, best_acc_epoch_miou, best_acc_epoch_avg_acc_miou))

#     if test_bbox_miou >= best_miou:
#         best_miou = test_bbox_miou
#         best_miou_epoch = epoch
#         best_miou_epoch_acc = test_acc
#         best_miou_epoch_fscore = test_fscore
#         best_miou_epoch_avg_acc_miou = test_avg_acc_miou
    
#     print('Best mIoU epoch:  %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_miou_epoch, best_miou_epoch_acc, best_miou_epoch_fscore, best_miou, best_miou_epoch_avg_acc_miou))         

#     if test_avg_acc_miou >= best_avg_acc_miou:
#         best_avg_acc_miou = test_avg_acc_miou
#         best_avg_acc_miou_epoch = epoch
#         best_avg_epoch_acc = test_acc
#         best_avg_epoch_fscore = test_fscore
#         best_avg_epoch_miou = test_bbox_miou

#         # checkpoint_path = args.checkpoint_dir + 'Best.pth'
#         # torch.save(model.state_dict(), checkpoint_path)
    
#     print('Best Avg epoch:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_avg_acc_miou_epoch, best_avg_epoch_acc, best_avg_epoch_fscore, best_avg_epoch_miou, best_avg_acc_miou))

#     return best_avg_acc_miou_epoch, best_avg_epoch_acc, best_avg_epoch_fscore, best_avg_epoch_miou, best_avg_acc_miou

class AvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is logits from Monte_carlo sampling of the model, true labels,
    and the type of uncertainty to be used [0: predictive uncertainty (default); 
    1: model uncertainty]
    """
    def __init__(self, beta=1):
        super(AvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def expected_entropy(self, mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=0)

    def predictive_uncertainty(self, mc_preds):
        """
        Compute the entropy of the mean of the predictive distribution
        obtained from Monte Carlo sampling.
        """
        return self.entropy(torch.mean(mc_preds, dim=0))

    def model_uncertainty(self, mc_preds):
        """
        Compute the difference between the entropy of the mean of the
        predictive distribution and the mean of the entropy.
        """
        return self.entropy(torch.mean(
            mc_preds, dim=0)) - self.expected_entropy(mc_preds)

    def accuracy_vs_uncertainty(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        # number of samples accurate and certain
        n_ac = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and certain
        n_ic = torch.zeros(1, device=true_label.device)
        # number of samples accurate and uncertain
        n_au = torch.zeros(1, device=true_label.device)
        # number of samples inaccurate and uncertain
        n_iu = torch.zeros(1, device=true_label.device)

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)

        for i in range(len(true_label)):
            if ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                """ accurate and certain """
                n_ac += 1
            elif ((true_label[i].item() == prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ accurate and uncertain """
                n_au += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() <= optimal_threshold):
                """ inaccurate and certain """
                n_ic += 1
            elif ((true_label[i].item() != prediction[i].item())
                  and uncertainty[i].item() > optimal_threshold):
                """ inaccurate and uncertain """
                n_iu += 1

        print('n_ac: ', n_ac, ' ; n_au: ', n_au, ' ; n_ic: ', n_ic, ' ;n_iu: ',
              n_iu)
        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)

        return avu

    def forward(self, logits, labels, optimal_uncertainty_threshold, type=0):

        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)

        if type == 0:
            unc = self.entropy(probs)
        else:
            unc = self.model_uncertainty(probs)

        unc_th = torch.tensor(optimal_uncertainty_threshold,
                              device=logits.device)

        n_ac = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=logits.device)
        avu_loss = torch.zeros(1, device=logits.device)

        for i in range(len(labels)):
            if ((labels[i].item() == predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                """ accurate and certain """
                n_ac += confidences[i] * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() == predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ accurate and uncertain """
                n_au += confidences[i] * torch.tanh(unc[i])
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() <= unc_th.item()):
                """ inaccurate and certain """
                n_ic += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
            elif ((labels[i].item() != predictions[i].item())
                  and unc[i].item() > unc_th.item()):
                """ inaccurate and uncertain """
                n_iu += (1 - confidences[i]) * torch.tanh(unc[i])

        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        p_ac = (n_ac) / (n_ac + n_ic)
        p_ui = (n_iu) / (n_iu + n_ic)
        #print('Actual AvU: ', self.accuracy_vs_uncertainty(predictions, labels, uncertainty, optimal_threshold))
        avu_loss = -1 * self.beta * torch.log(avu + self.eps)
        return avu_loss


def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)