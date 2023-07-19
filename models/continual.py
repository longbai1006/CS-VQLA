import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from transformers import BertTokenizer
from utils.vqla import *
from models.evaluation import test_endovis18, test_endovis17, test_mi2cai, test_endovis18_and_17, test_all_three_datasets
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class CSVQLA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self.old_network_module_ptr = self._old_network

    def incremental_train(self, task_id,
                train_loader_step0, 
                train_loader_step1,
                train_loader_step2,
                test_loader_step0,
                test_loader_step1,
                test_loader_step2):
        self._network_module_ptr = self._network
    

        if task_id == 0:
            self._total_classes = 26

        elif task_id == 1:
            self._total_classes = 28
            self._known_classes = 26

        elif task_id == 2:
            self._total_classes = 33
            self._known_classes = 28

        self._network.update_fc(self._total_classes)
        
        self._train(task_id,
                train_loader_step0, 
                train_loader_step1,
                train_loader_step2,
                test_loader_step0,
                test_loader_step1,
                test_loader_step2)

    def _train(self, task_id,
                train_loader_step0, 
                train_loader_step1,
                train_loader_step2,
                test_loader_step0,
                test_loader_step1,
                test_loader_step2):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if task_id == 0:
            optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["init_lr"])
            self._init_train(train_loader_step0, test_loader_step0, optimizer)
        elif task_id == 1:
            optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lrate"])
            self._update_representation_step1(train_loader_step1, test_loader_step0, test_loader_step1, optimizer)
            self._network.weight_align(self._total_classes - self._known_classes)
        
        elif task_id == 2:
            optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lrate"])
            self._update_representation_step2(train_loader_step2, test_loader_step0, test_loader_step1, test_loader_step2, optimizer)
            self._network.weight_align(self._total_classes - self._known_classes)
    
    def _init_train(self, train_loader_0, test_loader_0, optimizer):
        print('#===========Step 0 Start===========#')
        best_avg_step0 = 0
        best_epoch_step0 = 0

        for _, epoch in enumerate(range(self.args["init_epoch"])):
            vqla_loss = 0.0    
            vqla_loss_class = 0.0
            vqla_loss_bbox = 0.0  
            answering_label_true = None
            answering_label_pred = None
            bbox_outputs_pred = None
            bbox_label_true = None
            
            criterion = nn.CrossEntropyLoss()
            epochs_since_improvement = 0

            self._network.train()
            
            for i, (_, visual_features, q, labels, bbox_label) in enumerate(train_loader_0,0):
                
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

                loss_class = criterion(classification_outputs, labels)
                loss_bbox = loss_giou_l1(bbox_outputs, bbox_label)  
                loss = loss_class + loss_bbox        
                
                # zero the parameter gradients
                optimizer.zero_grad()        
                loss.backward()
                optimizer.step()

                # print statistics
                vqla_loss += loss.item()
                vqla_loss_class += loss_class.item()
                vqla_loss_bbox += loss_bbox.item()
                
                scores, predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
                answering_label_true = labels.data.cpu() if answering_label_true == None else torch.cat((answering_label_true, labels.data.cpu()), 0)
                answering_label_pred = predicted.data.cpu() if answering_label_pred == None else torch.cat((answering_label_pred, predicted.data.cpu()), 0)
                bbox_outputs_pred = bbox_outputs.data.cpu() if bbox_outputs_pred == None else torch.cat((bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
                bbox_label_true = bbox_label.data.cpu() if bbox_label_true == None else torch.cat((bbox_label_true, bbox_label.data.cpu()), 0)
                
            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)    
            
            print('\n\n Test on EndoVis18:')        
            _, _, _, _, step0_avg_step0 = test_endovis18(self, test_loader_0, optimizer, epoch)

            if step0_avg_step0 >= best_avg_step0:
                best_avg_step0 = step0_avg_step0
                torch.save(self._network.state_dict(), self.args["checkpoint_dir"] + 'step0.pth')

        print('\n\n Step 0 Best Epoch:')
        self._network.load_state_dict(torch.load(self.args["checkpoint_dir"] + 'step0.pth'))
        self._network.eval()
        print('Test on EndoVis18:')
        _, _, _, _, step0_avg_step0 = test_endovis18(self, test_loader_0, optimizer, epoch)

        print('#===========Step 0 End===========#')
    
    def _update_representation_step1(self, train_loader_1, test_loader_0, test_loader_1, optimizer):

        print('\n\n\n #===========Step 1 Start===========#')
        best_avg_both = 0
        for _, epoch in enumerate(range(self.args["epochs"])):
            self._network.train()

            vqla_loss = 0.0    
            answering_label_true = None
            answering_label_pred = None
            bbox_outputs_pred = None
            bbox_label_true = None
            epochs_since_improvement = 0

            for i, (_, visual_features, q, labels, bbox_label) in enumerate(train_loader_1,0):
                
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


                criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
                loss_clf = 100 * criterion(classification_outputs, labels)

                features = self._network_module_ptr.extract_vector(inputs, visual_features)
                features = features.view(-1,8,16,16)
                features = self._network_module_ptr.scpa_path(features)
                features = features.view(-1,2048)
                features_old = self.old_network_module_ptr.extract_vector(inputs, visual_features)
                loss_fkd = 10 * torch.dist(features, features_old, 2)
                (output1_old, _) = self._old_network(inputs, visual_features)
                _, output1_logits = torch.max(F.softmax(output1_old["logits"], dim=1).data, 1)    
                
                fake_classification = classification_outputs[:, :self._known_classes]
                fake_classification_old, output1_logits_old = extract_logits([23,24,25], fake_classification, output1_logits)
                fake_classification_overlap, output1_logits_overlap = extract_logits([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22], fake_classification, output1_logits)
                loss_kd_old = loss_kd_regularization(fake_classification_old, output1_logits_old, self.args["T_old"], 1)
                loss_kd_overlap = loss_kd_regularization(fake_classification_overlap, output1_logits_overlap, self.args["T_overlap"], 1)
                loss_kd = loss_kd_old + loss_kd_overlap
                
                loss_class = loss_clf + loss_kd + loss_fkd
                loss_bbox = loss_giou_l1(bbox_outputs, bbox_label)  
                loss = loss_class + loss_bbox        

                # zero the parameter gradients
                optimizer.zero_grad()        
                loss.backward()
                optimizer.step()

                # print statistics
                vqla_loss += loss.item()

                scores, predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
                answering_label_true = labels.data.cpu() if answering_label_true == None else torch.cat((answering_label_true, labels.data.cpu()), 0)
                answering_label_pred = predicted.data.cpu() if answering_label_pred == None else torch.cat((answering_label_pred, predicted.data.cpu()), 0)
                bbox_outputs_pred = bbox_outputs.data.cpu() if bbox_outputs_pred == None else torch.cat((bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
                bbox_label_true = bbox_label.data.cpu() if bbox_label_true == None else torch.cat((bbox_label_true, bbox_label.data.cpu()), 0)

            
            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8) 
            
            print('\n\n Test on EndoVis18:')
            _, _, _, _, step1_avg_step0 = test_endovis18(self, test_loader_0, optimizer, epoch)
            print('Test on EndoVis17:')
            _, _, _, _, step1_avg_step1 = test_endovis17(self, test_loader_1, optimizer, epoch)           
            print('Test on EndoVis18 & EndoVis17')
            _, _, _, _, step1_avg_both = test_endovis18_and_17(self, test_loader_0, test_loader_1, optimizer, epoch)           
            
            step1_result= step1_avg_step0 + step1_avg_step1
            if step1_result >= best_avg_both:
                best_avg_both = step1_result
                torch.save(self._network.state_dict(), self.args["checkpoint_dir"] + 'step1.pth')

        print('\n\n Step 1 Best Epoch:')
        self._network.load_state_dict(torch.load(self.args["checkpoint_dir"] + 'step1.pth'))
        self._network.eval()
        print('Test on EndoVis18:')
        _, _, _, _, step1_avg_step0 = test_endovis18(self, test_loader_0, optimizer, epoch)
        print('Test on EndoVis17:')
        _, _, _, _, step1_avg_step1 = test_endovis17(self, test_loader_1, optimizer, epoch)   
        print('Test on EndoVis18 & EndoVis17')
        _, _, _, _, step1_avg_both = test_endovis18_and_17(self, test_loader_0, test_loader_1, optimizer, epoch)         
        
        print('#===========Step 1 End===========#')

    def _update_representation_step2(self, train_loader_2, test_loader_0, test_loader_1, test_loader_2, optimizer):
        
        print('\n\n\n #===========Step 2 Start===========#')

        best_avg_all = 0
        for _, epoch in enumerate(range(self.args["epochs"])):
            self._network.train()
            vqla_loss = 0.0    
            answering_label_true = None
            answering_label_pred = None
            bbox_outputs_pred = None
            bbox_label_true = None
            
            epochs_since_improvement = 0

            for i, (_, visual_features, q, labels, bbox_label) in enumerate(train_loader_2,0):                
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
                criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
                loss_clf = 100 * criterion(classification_outputs, labels)

                features = self._network_module_ptr.extract_vector(inputs, visual_features)
                features = features.view(-1,8,16,16)
                features = self._network_module_ptr.scpa_path(features)
                features = features.view(-1,2048)
                features_old = self.old_network_module_ptr.extract_vector(inputs, visual_features)
                loss_fkd = 5 * torch.dist(features, features_old, 2)
                (output1_old, _) = self._old_network(inputs, visual_features)
                _, output1_logits = torch.max(F.softmax(output1_old["logits"], dim=1).data, 1) 
            
                fake_classification = classification_outputs[:, :self._known_classes]
                fake_classification_old, output1_logits_old = extract_logits([3,5,8,9,12,13,20,21,22,23,24,25,26,27], fake_classification, output1_logits)
                fake_classification_overlap, output1_logits_overlap = extract_logits([0,1,2,4,6,7,10,11,14,15,16,17,18,19], fake_classification, output1_logits)
                loss_kd_old = loss_kd_regularization(fake_classification_old, output1_logits_old, self.args["T_old"], 1)
                loss_kd_overlap = loss_kd_regularization(fake_classification_overlap, output1_logits_overlap, self.args["T_overlap"], 1)
                loss_kd = loss_kd_old + loss_kd_overlap
                
                loss_class = loss_clf + loss_kd + loss_fkd
                loss_bbox = loss_giou_l1(bbox_outputs, bbox_label)  
                loss = loss_class + loss_bbox        
                optimizer.zero_grad()        
                loss.backward()
                optimizer.step()

                # print statistics
                vqla_loss += loss.item()
                scores, predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
                answering_label_true = labels.data.cpu() if answering_label_true == None else torch.cat((answering_label_true, labels.data.cpu()), 0)
                answering_label_pred = predicted.data.cpu() if answering_label_pred == None else torch.cat((answering_label_pred, predicted.data.cpu()), 0)
                bbox_outputs_pred = bbox_outputs.data.cpu() if bbox_outputs_pred == None else torch.cat((bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
                bbox_label_true = bbox_label.data.cpu() if bbox_label_true == None else torch.cat((bbox_label_true, bbox_label.data.cpu()), 0)
                
            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)            
        
            print('\n\n Test on EndoVis18:')
            _, _, _, _, step2_avg_step0 = test_endovis18(self, test_loader_0, optimizer, epoch)
            print('Test on EndoVis17:')
            _, _, _, _, step2_avg_step1 = test_endovis17(self, test_loader_1, optimizer, epoch)           
            print('Test onS M2CAI:')
            _, _, _, _, step2_avg_step2 = test_mi2cai(self, test_loader_2, optimizer, epoch)           
            print('Test on EndoVis18 & EndoVis17')
            _, _, _, _, step2_avg_both = test_endovis18_and_17(self, test_loader_0, test_loader_1, optimizer, epoch)           
            print('Test on Final')
            _, _, _, _, step2_avg_all = test_all_three_datasets(self, test_loader_0, test_loader_1, test_loader_2, optimizer, epoch)           

            step2_result = step2_avg_step0 + step2_avg_step1 + step2_avg_step2
            if step2_result >= best_avg_all:
                best_avg_all = step2_result
                torch.save(self._network.state_dict(), self.args["checkpoint_dir"] + 'step2.pth')

        print('\n\n Step 2 Best Epoch:')
        self._network.load_state_dict(torch.load(self.args["checkpoint_dir"] + 'step2.pth'))
        self._network.eval()
        print('Test on EndoVis18:')
        _, _, _, _, step2_avg_step0 = test_endovis18(self, test_loader_0, optimizer, epoch)
        print('Test on EndoVis17:')
        _, _, _, _, step2_avg_step1 = test_endovis17(self, test_loader_1, optimizer, epoch)           
        print('Test on M2CAI:')
        _, _, _, _, step2_avg_step2 = test_mi2cai(self, test_loader_2, optimizer, epoch)              
        print('Test on EndoVis18 & EndoVis17')
        _, _, _, _, step2_avg_both = test_endovis18_and_17(self, test_loader_0, test_loader_1, optimizer, epoch)           
        print('Test on Final')
        _, _, _, _, step2_avg_all = test_all_three_datasets(self, test_loader_0, test_loader_1, test_loader_2, optimizer, epoch)           


        print('#===========Step 2 End===========#')

def loss_kd_regularization(outputs, labels, T, multiplier):
    """
    loss function for mannually-designed regularization: Tf-KD_{reg}
    """
    correct_prob = 0.9    # the probability for correct class in u(k)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft*(1-correct_prob)/(K-1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i ,labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_soft/T, dim=1)) * multiplier
    return loss_soft_regu


def extract_logits(class_list_name, fake_classification, output1_logits): 
    
    full_fake_target = None
    full_fake_output = None
    
    for i in class_list_name:
        index_tool = torch.argwhere(output1_logits == i).squeeze()
        fake_target = torch.index_select(output1_logits, 0, index_tool)
        fake_output = torch.index_select(fake_classification, 0, index_tool)

        full_fake_target = fake_target if full_fake_target == None else torch.cat((full_fake_target, fake_target), 0)
        full_fake_output = fake_output if full_fake_output == None else torch.cat((full_fake_output, fake_output), 0)

    return full_fake_output, full_fake_target