from __future__ import division
import lib
import sys

class Evaluator(object):
    def __init__(self, model, metrics, dicts, opt):
        self.model = model
        self.loss_func = metrics["xent_loss"]
        self.sent_reward_func = metrics["sent_reward"]
        self.corpus_reward_func = metrics["corp_reward"]
        self.dicts = dicts
        self.max_length = opt.max_predict_length
        self.opt = opt

    def eval(self, data, pred_file=None):
        self.model.eval()

        total_loss = 0
        total_words = 0
        total_sents = 0
        total_sent_reward = 0

        all_preds = []
        all_targets = []
        all_srcs = []
        for i in range(len(data)): #
        # for i in range(7818, 7819):
            batch = data[i]
            # print "convertToLabels: ", self.dicts
            # print self.dicts["tgt"].convertToLabels(batch[2], lib.Constants.EOS)
            # print("tgt-batch[2]: ")
            # t = [a[0] for a in batch[2].data.cpu().numpy().tolist()]
            # t_labels = self.dicts["tgt"].convertToLabels(t, lib.Constants.EOS)
            # print(' '.join(t_labels[:-1]).encode('utf-8', 'ignore'))

            # print("code-batch[1][2][0].data")
            # print(batch[1][2][0].data.cpu().numpy().tolist())
            # code = [a[0] for a in batch[1][2][0].data.cpu().numpy().tolist()]
            # code_labels = self.dicts["src"].convertToLabels(code, lib.Constants.EOS)
            # print(' '.join(code_labels[:-1]).encode('utf-8', 'ignore'))

            # print("txt-batch[0][0].data")
            # print(batch[0][0].data.cpu().numpy().tolist())
            # txt_labels = []
            # for i in range(len(batch[0][0].data.cpu().numpy().tolist()[0])):
            #     txt = [a[i] for a in batch[0][0].data.cpu().numpy().tolist()]
            #     txt_lbl = self.dicts["src"].convertToLabels(txt, lib.Constants.EOS)
            #     txt_labels.append(' '.join(txt_lbl[:-1]).encode('utf-8', 'ignore'))
            # print('txt_labels: ')
            # print(txt_labels)
            # print(' '.join(txt_labels[:-1]).encode('utf-8', 'ignore'))

            if self.opt.data_type == 'code':
                targets = batch[2]
                # attention_mask = batch[2][0].data.eq(lib.Constants.PAD).t()
                attention_mask = batch[1][2][0].data.eq(lib.Constants.PAD).t()
            elif self.opt.data_type == 'text':
                targets = batch[2]
                # attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
                attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
            elif self.opt.data_type == 'hybrid':
                targets = batch[2]
                attention_mask_code = batch[1][2][0].data.eq(lib.Constants.PAD).t()
                attention_mask_txt = batch[0][0].data.eq(lib.Constants.PAD).t()
                # print "targets: "
                # print targets
                # print "batch[1][2][0]: "
                # print batch[1][2][0]
                # print "src_lengths: "
                # print batch[0][1]
                # print "tree_lengths: "
                # print batch[1][1]
                # print "attention_mask_code: "
                # print attention_mask_code
                # print "attention_mask_txt: "
                # print attention_mask_txt

            if self.opt.has_attn:
                if self.opt.data_type == 'code' or self.opt.data_type == 'text':
                    self.model.decoder.attn.applyMask(attention_mask)
                elif self.opt.data_type == 'hybrid':
                    self.model.decoder.attn.applyMask(attention_mask_code, attention_mask_txt)

            outputs = self.model(batch, True)

            weights = targets.ne(lib.Constants.PAD).float()
            num_words = weights.data.sum()
            _, loss = self.model.predict(outputs, targets, weights, self.loss_func)

            preds = self.model.translate(batch, self.max_length)
            # print('preds: ')
            # print(preds)
            preds = preds.t().tolist()
            # print('preds-: ')
            # print(preds)
            srcs = batch[0][0]
            # print('srcs: ')
            # print(srcs)
            # print('srcs-: ')
            srcs = srcs.data.t().tolist()
            # print(srcs)
            # print("targets:")
            # print(targets)
            targets = targets.data.t().tolist()
            # print("targets-:")
            # print(targets)

            rewards, _ = self.sent_reward_func(preds, targets)

            # sent = [self.dicts["tgt"].getLabel(w) for w in preds]
            # tgt = [self.dicts["tgt"].getLabel(w) for w in targets]
            # print "sent: ", " ".join(sent)
            # print "tgt: ", " ".join(tgt)

            all_preds.extend(preds)
            all_targets.extend(targets)
            all_srcs.extend(srcs)

            total_loss += loss
            total_words += num_words
            total_sent_reward += sum(rewards)

            if self.opt.data_type == 'code':
                total_sents += batch[2].size(1)
            elif self.opt.data_type == 'text':
                total_sents += batch[2].size(1)
            elif self.opt.data_type == 'hybrid':
                total_sents += batch[2].size(1)

        loss = total_loss / total_words
        sent_reward = total_sent_reward / total_sents
        corpus_reward = self.corpus_reward_func(all_preds, all_targets)
        # print "all_0: "
        # print all_preds[0]
        # print all_targets[0]
        if pred_file is not None:
            self._convert_and_report(data, pred_file, all_preds, all_targets, all_srcs, (loss, sent_reward, corpus_reward))

        return loss, sent_reward, corpus_reward

    def _convert_and_report(self, data, pred_file, preds, targets, srcs, metrics):
        # preds = data.restore_pos(preds)
        with open(pred_file, "w") as f:
            for i in range(len(preds)):
                pred = preds[i]
                target = targets[i]
                src = srcs[i]

                src = lib.Reward.clean_up_sentence(src, remove_unk=False, remove_eos=True)
                pred = lib.Reward.clean_up_sentence(pred, remove_unk=False, remove_eos=True)
                target = lib.Reward.clean_up_sentence(target, remove_unk=False, remove_eos=True)

                src = [self.dicts["src"].getLabel(w) for w in src]
                pred = [self.dicts["tgt"].getLabel(w) for w in pred]
                tgt = [self.dicts["tgt"].getLabel(w) for w in target]

                f.write(str(i) + ": src: "+ " ".join(src).encode('utf-8', 'ignore') + '\n')
                f.write(str(i) + ": pre: " + " ".join(pred).encode('utf-8', 'ignore') + '\n')
                f.write(str(i) + ": tgt: "+ " ".join(tgt).encode('utf-8', 'ignore') + '\n')

        loss, sent_reward, corpus_reward = metrics
        print("")
        print("Loss: %.6f" % loss)
        print("Sentence reward: %.2f" % (sent_reward * 100))
        print("Corpus reward: %.2f" % (corpus_reward * 100))
        print("Predictions saved to %s" % pred_file)


