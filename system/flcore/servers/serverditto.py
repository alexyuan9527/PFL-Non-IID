import copy
import numpy as np
import time
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread


class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDitto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    # train()在不同算法中有所不同，因此在特定的 server 类中定义
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            # evaluate 两次，h5 文件中 loss 和 acc 中偶数序号是 global model，奇数序号是 personalized model
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate()

            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate_personalized()

            for client in self.selected_clients:
                if client.malicious:
                    client.mptrain()
                    client.gauss_attack()
                else:
                    client.ptrain()
                    client.train()

            # threads = [Thread(target=client.train) for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:  # DLG
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDitto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        benign_clients = self.clients[self.mcnum:]
        for c in benign_clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in benign_clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        benign_clients = self.clients[self.mcnum:]
        for c in benign_clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in benign_clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate_personalized(self, acc=None, loss=None, acc_std=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        acc_ls = [a / n for a, n in zip(stats[2], stats[1])]
        accs = np.std(acc_ls)
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if acc_std == None:
            self.rs_test_acc_std.append(accs)
        else:
            acc_std.append(accs)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc), "   Std Test Accurancy: {:.4f}".format(accs))
        # print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test Accurancy: {:.4f}".format(accs))
        # print("Std Test AUC: {:.4f}".format(np.std(aucs)))