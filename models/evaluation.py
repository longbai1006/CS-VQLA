from utils.vqla import *
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def test_endovis18(self, test_loader, optimizer, epoch):
    
    self._network.eval()

    val_answering_label_true = None
    val_answering_label_pred = None
    val_bbox_outputs_pred = None
    val_bbox_label_true = None

    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader, 0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]
        
        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)

    acc = calc_acc(val_answering_label_true, val_answering_label_pred) 
    c_acc = 0.0
    precision, recall, fscore = calc_precision_recall_fscore(val_answering_label_true, val_answering_label_pred)
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(val_bbox_label_true), box_cxcywh_to_xyxy(val_bbox_outputs_pred))    
    print('Test: epoch: %d | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f' %(epoch, acc, precision, recall, fscore, bbox_miou))

    acc_18_unique, miou_18_unique = class_wise_result_range(23, 25,val_answering_label_true, val_answering_label_pred, val_bbox_outputs_pred, val_bbox_label_true)     
    print('18 non-overlap          : Acc: %.6f | mIoU: %.6f |' %(acc_18_unique, miou_18_unique))

    return epoch, acc, fscore, bbox_miou, 0.5 * (acc + bbox_miou)

def test_mi2cai(self, test_loader, optimizer, epoch):

    self._network.eval()
    
    val_answering_label_true = None
    val_answering_label_pred = None
    val_bbox_outputs_pred = None
    val_bbox_label_true = None

    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader,0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]

        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)
    
    acc = calc_acc(val_answering_label_true, val_answering_label_pred) 
    c_acc = 0.0
    precision, recall, fscore = calc_precision_recall_fscore(val_answering_label_true, val_answering_label_pred)
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(val_bbox_label_true), box_cxcywh_to_xyxy(val_bbox_outputs_pred))    
    print('Test: epoch: %d | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f' %(epoch, acc, precision, recall, fscore, bbox_miou))

    acc_mi2cai_unique, miou_mi2cai_unique = class_wise_result_range(28, 32, val_answering_label_true, val_answering_label_pred, val_bbox_outputs_pred, val_bbox_label_true)     
    print('mi2cai non-overlap: Acc: %.6f | mIoU: %.6f |' %(acc_mi2cai_unique, miou_mi2cai_unique))

    return epoch, acc, fscore, bbox_miou, 0.5 * (acc + bbox_miou)

def test_endovis17(self, test_loader, optimizer, epoch):

    self._network.eval()
    
    val_answering_label_true = None
    val_answering_label_pred = None
    val_bbox_outputs_pred = None
    val_bbox_label_true = None

    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader,0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]

        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)
    
    acc = calc_acc(val_answering_label_true, val_answering_label_pred) 
    c_acc = 0.0
    precision, recall, fscore = calc_precision_recall_fscore(val_answering_label_true, val_answering_label_pred)
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(val_bbox_label_true), box_cxcywh_to_xyxy(val_bbox_outputs_pred))    
    print('Test: epoch: %d | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f' %(epoch, acc, precision, recall, fscore, bbox_miou))

    acc_17_unique, miou_17_unique = class_wise_result_range(26, 27, val_answering_label_true, val_answering_label_pred, val_bbox_outputs_pred, val_bbox_label_true)     
    print('17 non-overlap          : Acc: %.6f | mIoU: %.6f |' %(acc_17_unique, miou_17_unique))

    return epoch, acc, fscore, bbox_miou, 0.5 * (acc + bbox_miou)

def test_endovis18_and_17(self, test_loader_18, test_loader_17, optimizer, epoch):

    self._network.eval()
    
    val_answering_label_true = None
    val_answering_label_pred = None
    val_bbox_outputs_pred = None
    val_bbox_label_true = None

    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader_18,0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]

        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)
    
    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader_17,0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]

        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)
    
    acc = calc_acc(val_answering_label_true, val_answering_label_pred) 
    c_acc = 0.0
    precision, recall, fscore = calc_precision_recall_fscore(val_answering_label_true, val_answering_label_pred)
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(val_bbox_label_true), box_cxcywh_to_xyxy(val_bbox_outputs_pred))    
    print('Test: epoch: %d | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f' %(epoch, acc, precision, recall, fscore, bbox_miou))

    acc_1718_overlap, miou_1718_overlap = class_wise_result_range(0, 22, val_answering_label_true, val_answering_label_pred, val_bbox_outputs_pred, val_bbox_label_true)
    print('18 & 17 overlap         : Acc: %.6f | mIoU: %.6f |' %(acc_1718_overlap, miou_1718_overlap))

    acc_1718_unique, miou_1718_unique = class_wise_specific([3,5,8,9,12,13,20,21,22,23,24,25,26,27], val_answering_label_true, val_answering_label_pred, val_bbox_outputs_pred, val_bbox_label_true)  
    print('18 & 17 unique          : Acc: %.6f | mIoU: %.6f |' %(acc_1718_unique, miou_1718_unique))

    return epoch, acc, fscore, bbox_miou, 0.5 * (acc + bbox_miou)

def test_all_three_datasets(self, test_loader_18, test_loader_17, test_loader_mi2cai, optimizer, epoch):

    self._network.eval()
    
    val_answering_label_true = None
    val_answering_label_pred = None
    val_bbox_outputs_pred = None
    val_bbox_label_true = None

    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader_18,0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]

        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)
    
    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader_17,0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]

        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)

    for i, (_, visual_features, q, labels, bbox_label) in enumerate(test_loader_mi2cai,0):
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)

        # GPU / CPU
        visual_features = visual_features.to(self._device)
        labels = labels.to(self._device)
        bbox_label = bbox_label.to(self._device)

        (output1, output2) = self._network(inputs, visual_features)
        classification_outputs = output1["logits"]
        bbox_outputs = output2["bbox"]

        scores, val_predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        val_answering_label_true = labels.data.cpu() if val_answering_label_true == None else torch.cat((val_answering_label_true, labels.data.cpu()), 0)
        val_answering_label_pred = val_predicted.data.cpu() if val_answering_label_pred == None else torch.cat((val_answering_label_pred, val_predicted.data.cpu()), 0)
        val_bbox_outputs_pred = bbox_outputs.data.cpu() if val_bbox_outputs_pred == None else torch.cat((val_bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        val_bbox_label_true = bbox_label.data.cpu() if val_bbox_label_true == None else torch.cat((val_bbox_label_true, bbox_label.data.cpu()), 0)
       
    acc = calc_acc(val_answering_label_true, val_answering_label_pred) 
    c_acc = 0.0
    precision, recall, fscore = calc_precision_recall_fscore(val_answering_label_true, val_answering_label_pred)
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(val_bbox_label_true), box_cxcywh_to_xyxy(val_bbox_outputs_pred))    
    print('Test: epoch: %d | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f' %(epoch, acc, precision, recall, fscore, bbox_miou))

    acc_overlap, miou_overlap = class_wise_specific([0,1,2,4,6,7,10,11,14,15,16,17,18,19], val_answering_label_true, val_answering_label_pred, val_bbox_outputs_pred, val_bbox_label_true)
    print('18 & 17 & mi2cai overlap: Acc: %.6f | mIoU: %.6f |' %(acc_overlap, miou_overlap))

    return epoch, acc, fscore, bbox_miou, 0.5 * (acc + bbox_miou)

def class_wise_result_range(tool_id_down, tool_id_up,
                      label_true,
                      label_pred,
                      bbox_pred,
                      bbox_true): 
    
    full_label_true = None
    full_label_pred = None
    full_bbox_pred = None
    full_bbox_true = None
    
    for i in range(tool_id_down, tool_id_up+1):
        index_tool = torch.argwhere(label_true == i).squeeze()
        label_true_tool = torch.index_select(label_true, 0, index_tool)
        label_pred_tool = torch.index_select(label_pred, 0, index_tool)
        bbox_pred_tool = torch.index_select(bbox_pred, 0, index_tool)
        bbox_true_tool = torch.index_select(bbox_true, 0, index_tool)

        full_label_true = label_true_tool.data.cpu() if full_label_true == None else torch.cat((full_label_true, label_true_tool.data.cpu()), 0)
        full_label_pred = label_pred_tool.data.cpu() if full_label_pred == None else torch.cat((full_label_pred, label_pred_tool.data.cpu()), 0)
        full_bbox_pred = bbox_pred_tool.data.cpu() if full_bbox_pred == None else torch.cat((full_bbox_pred, bbox_pred_tool.data.cpu()), 0)
        full_bbox_true = bbox_true_tool.data.cpu() if full_bbox_true == None else torch.cat((full_bbox_true, bbox_true_tool.data.cpu()), 0)

    acc = calc_acc(full_label_true, full_label_pred) 
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(full_bbox_true), box_cxcywh_to_xyxy(full_bbox_pred))  

    return (acc, bbox_miou)


def class_wise_specific(class_list_name, label_true, label_pred, bbox_pred, bbox_true): 
    
    full_label_true = None
    full_label_pred = None
    full_bbox_pred = None
    full_bbox_true = None
    
    for i in class_list_name:
        index_tool = torch.argwhere(label_true == i).squeeze()
        label_true_tool = torch.index_select(label_true, 0, index_tool)
        label_pred_tool = torch.index_select(label_pred, 0, index_tool)
        bbox_pred_tool = torch.index_select(bbox_pred, 0, index_tool)
        bbox_true_tool = torch.index_select(bbox_true, 0, index_tool)

        full_label_true = label_true_tool.data.cpu() if full_label_true == None else torch.cat((full_label_true, label_true_tool.data.cpu()), 0)
        full_label_pred = label_pred_tool.data.cpu() if full_label_pred == None else torch.cat((full_label_pred, label_pred_tool.data.cpu()), 0)
        full_bbox_pred = bbox_pred_tool.data.cpu() if full_bbox_pred == None else torch.cat((full_bbox_pred, bbox_pred_tool.data.cpu()), 0)
        full_bbox_true = bbox_true_tool.data.cpu() if full_bbox_true == None else torch.cat((full_bbox_true, bbox_true_tool.data.cpu()), 0)

    acc = calc_acc(full_label_true, full_label_pred) 
    bbox_miou = mIoU_xyxy(box_cxcywh_to_xyxy(full_bbox_true), box_cxcywh_to_xyxy(full_bbox_pred))  

    return (acc, bbox_miou)