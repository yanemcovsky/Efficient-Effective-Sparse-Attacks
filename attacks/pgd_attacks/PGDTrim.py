import numpy as np
import math
import itertools
import torch
from attacks.pgd_attacks.attack import Attack


class PGDTrim(Attack):
    def __init__(
            self,
            model,
            criterion,
            misc_args=None,
            pgd_args=None,
            dropout_args=None,
            trim_args=None):

        super(PGDTrim, self).__init__(model, criterion, misc_args, pgd_args, dropout_args)

        self.name = "PGDTrim"
        self.sparsity = trim_args['sparsity']
        self.max_trim_steps = trim_args['max_trim_steps']
        self.trim_steps = trim_args['trim_steps']
        self.scale_dpo_mean = trim_args['scale_dpo_mean']
        if self.scale_dpo_mean:
            self.dropout_str += ", scaled by trim ratio"
        self.mask_dist_str = "multinomial"
        self.n_mask_samples = trim_args['n_mask_samples']
        self.l0_norms = None
        self.n_l0_norms = None
        self.sample_mask_pixels = None
        self.mask_prep = None
        self.mask_dist = None
        self.mask_sample = None
        self.post_trim_dpo_mean = None
        self.post_trim_dpo_std = None
        self.compute_trim_args(trim_args)

        self.restarts_trim_steps = None
        self.restarts_l0_indices = None
        self.restarts_l0_copy_indices = None
        self.restarts_trim_steps_ratios = None
        self.restarts_dpo_mean_steps = None
        self.restarts_dpo_std_steps = None
        self.restarts_mask_prep = None
        self.restarts_mask_sample = None
        self.restarts_n_mask_samples = None
        self.restarts_mask_compute = None
        self.compute_restarts_trim_steps()

        if self.report_info:
            self.output_l0_norms = self.l0_norms
        else:
            self.output_l0_norms = [self.sparsity]

    def compute_trim_args(self, trim_args):

        self.mask_sample = self.mask_sample_multinomial
        self.mask_prep = self.mask_prep_multinomial
        self.sample_mask_pixels = self.sample_mask_pixels_known_count

        if trim_args['post_trim_dpo']:
            self.post_trim_dpo_mean = self.dpo_mean
            self.post_trim_dpo_std = self.dpo_std
        else:
            self.post_trim_dpo_mean = torch.zeros_like(self.dpo_mean)
            self.post_trim_dpo_std = torch.zeros_like(self.dpo_std)

    def compute_trim_steps(self, trim_steps):
        curr_norm_ls = [self.n_data_pixels] + trim_steps[:-1]
        curr_norm = torch.tensor(curr_norm_ls, dtype=self.dtype, device=self.device)
        next_norm = torch.tensor(trim_steps, dtype=self.dtype, device=self.device)
        trim_steps_ratios = next_norm / curr_norm
        if self.scale_dpo_mean:
            dpo_mean_steps = self.dpo_mean * trim_steps_ratios
        else:
            dpo_mean_steps = self.dpo_mean.expand(len(trim_steps)).to(dtype=self.dtype, device=self.device)
        dpo_std_steps = self.compute_dpo_std(dpo_mean_steps)

        mask_prep = [self.mask_prep] * len(trim_steps)
        mask_sample = [self.mask_sample] * len(trim_steps)
        n_mask_samples = [self.n_mask_samples] * len(trim_steps)
        mask_compute = [self.mask_compute_best_pixels_crit] * len(trim_steps)
        comb_size_ls = [math.comb(curr, next) for curr, next in zip(curr_norm_ls, trim_steps)]
        sample_all_masks_ind = [idx for idx, comb_size in enumerate(comb_size_ls) if comb_size <= self.n_mask_samples]
        for idx in sample_all_masks_ind:
            mask_prep[idx] = self.mask_prep_comb
            mask_sample[idx] = self.mask_sample_comb
            n_mask_samples[idx] = comb_size_ls[idx]

        return trim_steps_ratios, dpo_mean_steps, dpo_std_steps, mask_prep, mask_sample, n_mask_samples, mask_compute

    def compute_restarts_trim_steps(self):
        if self.trim_steps is not None:
            self.sparsity = None
            self.max_trim_steps = None
            self.l0_norms = [self.n_data_pixels] + self.trim_steps
            self.n_l0_norms = len(self.l0_norms)

            trim_steps_ratios, dpo_mean_steps, dpo_std_steps, mask_prep, mask_sample, n_mask_samples, mask_compute\
                = self.compute_trim_steps(self.trim_steps)
            self.restarts_trim_steps = [self.trim_steps] * self.n_restarts
            self.restarts_l0_indices = [list(range(self.n_l0_norms))] * self.n_restarts
            self.restarts_l0_copy_indices = [[]] * self.n_restarts
            self.restarts_trim_steps_ratios = [trim_steps_ratios] * self.n_restarts
            self.restarts_dpo_mean_steps = [dpo_mean_steps] * self.n_restarts
            self.restarts_dpo_std_steps = [dpo_std_steps] * self.n_restarts
            self.restarts_mask_prep = [mask_prep] * self.n_restarts
            self.restarts_mask_sample = [mask_sample] * self.n_restarts
            self.restarts_n_mask_samples = [n_mask_samples] * self.n_restarts
            self.restarts_mask_compute = [mask_compute] * self.n_restarts
            return

        pixels_log_size = int(np.log2(self.n_data_pixels))
        max_trim_size = 2 ** pixels_log_size
        if max_trim_size < self.n_data_pixels:
            n_trim_options = int(np.ceil(np.log2(max_trim_size / self.sparsity)))
            all_l0_norms = [self.n_data_pixels] + [max_trim_size >> step for step in range(n_trim_options)] + [self.sparsity]
        else:
            n_trim_options = int(np.ceil(np.log2(self.n_data_pixels / self.sparsity))) - 1
            all_l0_norms = [self.n_data_pixels >> step for step in range(n_trim_options + 1)] + [self.sparsity]
        sparsity_trim_idx = len(all_l0_norms) - 1
        if self.max_trim_steps > n_trim_options + 1:
            self.max_trim_steps = n_trim_options + 1
            if self.max_trim_steps == 0:
                self.l0_norms = [self.sparsity]
                self.n_l0_norms = 1
                self.restarts_l0_indices = [[0]] * self.n_restarts
                self.restarts_trim_steps = [[]] * self.n_restarts
                self.restarts_l0_copy_indices = [[]] * self.n_restarts
                self.restarts_trim_steps_ratios = [[]] * self.n_restarts
                self.restarts_dpo_mean_steps = [[]] * self.n_restarts
                self.restarts_dpo_std_steps = [[]] * self.n_restarts
                self.restarts_mask_prep = [[]] * self.n_restarts
                self.restarts_mask_sample = [[]] * self.n_restarts
                self.restarts_n_mask_samples = [[]] * self.n_restarts
                self.restarts_mask_compute = [[]] * self.n_restarts
                return
        if self.n_restarts < self.max_trim_steps:
            if self.n_restarts == 1:
                step_size_list = [self.max_trim_steps]
            else:
                step_size_offset = (self.n_restarts - 1) - (self.max_trim_steps - 1) % (self.n_restarts - 1)
                step_size_list = [1] + [int((self.max_trim_steps - 1) / (self.n_restarts - 1)) + (i > step_size_offset) for i in range(1, self.n_restarts)]
            n_trim_steps_list = list(reversed(list(itertools.accumulate(step_size_list))))

            repeat = 1
        elif self.max_trim_steps == 0:
            n_trim_steps_list = []
            repeat = self.n_restarts
        else:
            n_trim_steps_list = list(reversed(list(range(1, self.max_trim_steps + 1))))
            repeat = int(np.ceil(self.n_restarts / self.max_trim_steps))

        all_steps_lists = []
        self.restarts_trim_steps = []
        l0_norm_is_computed = [1] + [0] * n_trim_options + [1]
        for n_trim_steps in n_trim_steps_list:

            step_size_offset = n_trim_steps - sparsity_trim_idx % n_trim_steps
            step_size_list = [int(sparsity_trim_idx / n_trim_steps) + (i > step_size_offset) for i in range(1, n_trim_steps)]
            steps_list = list(itertools.accumulate(step_size_list))
            trim_steps = [all_l0_norms[step] for step in steps_list] + [self.sparsity]
            self.restarts_trim_steps.append(trim_steps)
            for step in steps_list:
                l0_norm_is_computed[step] = 1
            all_steps_lists.append([0] + steps_list + [sparsity_trim_idx])

        self.l0_norms = [l0_norm for l0_norm_idx, l0_norm in enumerate(all_l0_norms) if l0_norm_is_computed[l0_norm_idx]]
        self.n_l0_norms = len(self.l0_norms)
        steps_l0_indices = list(itertools.accumulate(l0_norm_is_computed, initial=0))
        self.restarts_l0_indices = [[steps_l0_indices[step] for step in steps_list] for steps_list in all_steps_lists]

        restarts_l0_skip_indices = []
        for l0_indices in self.restarts_l0_indices:
            l0_skip_indices = []
            for prev_idx, next_idx in zip(l0_indices, l0_indices[1:]):
                l0_skip_indices.extend(list(range(prev_idx + 1, next_idx)))
            restarts_l0_skip_indices.append(l0_skip_indices)

        self.restarts_trim_steps = (self.restarts_trim_steps * repeat)[:self.n_restarts]
        self.restarts_l0_indices = (self.restarts_l0_indices * repeat)[:self.n_restarts]
        restarts_l0_skip_indices = (restarts_l0_skip_indices * repeat)[:self.n_restarts]

        self.restarts_l0_copy_indices = []
        l0_norm_prev_computed = [0] * self.n_l0_norms
        for rest_idx, l0_indices in enumerate(self.restarts_l0_indices):
            l0_skip_indices = restarts_l0_skip_indices[rest_idx]
            self.restarts_l0_copy_indices.append([l0_skip_idx for l0_skip_idx in l0_skip_indices if l0_norm_prev_computed[l0_skip_idx]])
            for l0_idx in l0_indices:
                l0_norm_prev_computed[l0_idx] = True

        self.restarts_trim_steps_ratios = []
        self.restarts_dpo_mean_steps = []
        self.restarts_dpo_std_steps = []
        self.restarts_mask_prep = []
        self.restarts_mask_sample = []
        self.restarts_n_mask_samples = []
        self.restarts_mask_compute = []
        for trim_steps in self.restarts_trim_steps:
            trim_steps_ratios, dpo_mean_steps, dpo_std_steps, mask_prep, mask_sample, n_mask_samples, mask_compute\
                = self.compute_trim_steps(trim_steps)
            self.restarts_trim_steps_ratios.append(trim_steps_ratios)
            self.restarts_dpo_mean_steps.append(dpo_mean_steps)
            self.restarts_dpo_std_steps.append(dpo_std_steps)
            self.restarts_mask_prep.append(mask_prep)
            self.restarts_mask_sample.append(mask_sample)
            self.restarts_n_mask_samples.append(n_mask_samples)
            self.restarts_mask_compute.append(mask_compute)
        return

    def report_schematics(self):

        print("Perturbations will be computed for the L0 norms:")
        print(self.l0_norms)
        print("The best performing perturbations will be reported for the L0 norms:")
        print(self.output_l0_norms)
        print("perturbations L_inf norm limitation:")
        print(self.eps_ratio)
        print("L0 trim steps schedule for the attack:")
        for rest_idx, trim_steps in enumerate(self.restarts_trim_steps):
            print("Attack restart index: " + str(rest_idx) + " L0 trim steps: " + str(trim_steps))
        print("Number of iterations for optimizing perturbations in each trim step:")
        print(self.n_iter)
        print("perturbations will be optimized with the dropout distribution:")
        print(self.dropout_str)

    def compute_mask_from_ind(self, mask_indices):
        return self.mask_zeros_flat.scatter(dim=1, index=mask_indices, src=self.mask_ones_flat).view(self.mask_shape)

    def mask_dist_multinomial(self, pixels_prob, n_trim_pixels_tensor):
        return torch.distributions.multinomial.Multinomial(total_count=n_trim_pixels_tensor, probs=pixels_prob)

    def mask_prep_multinomial(self, pixels_prob, n_trim_pixels_tensor):
        return pixels_prob, n_trim_pixels_tensor

    def mask_prep_comb(self, pixels_prob, n_trim_pixels_tensor):
        active_pixels_ind = pixels_prob.nonzero(as_tuple=True)[1].view(self.batch_size, -1)
        n_active_pixels = active_pixels_ind.shape[1]
        trim_ind_from_active = torch.arange(end=n_active_pixels, dtype=self.dtype, device=self.device)
        trim_comb = torch.combinations(trim_ind_from_active, r=n_trim_pixels_tensor).to(dtype=torch.int)
        return trim_comb, active_pixels_ind

    def mask_sample_multinomial(self, mask_prep, index):
        pixels_prob, n_trim_pixels_tensor = mask_prep
        return self.compute_mask_from_ind(pixels_prob.multinomial(n_trim_pixels_tensor, replacement=False))

    def mask_sample_comb(self, mask_prep, index):
        trim_comb, active_pixels_ind = mask_prep
        trim_ind_from_active = trim_comb[index]
        return self.compute_mask_from_ind(active_pixels_ind[:, trim_ind_from_active])

    def sample_mask_pixels_known_count(self, mask_prep, n_trim_pixels_tensor, sample_idx):
        mask_sample = self.mask_sample(mask_prep, sample_idx)
        return mask_sample, n_trim_pixels_tensor

    def mask_compute_best_pixels_crit(self, x, y, pert, n_trim_pixels_tensor, mask_prep_data, n_mask_samples):
        with torch.no_grad():
            pixel_sample_count = torch.zeros(self.mask_shape, dtype=self.dtype, device=self.device)
            pixel_loss_sum = torch.zeros(self.mask_shape, dtype=self.dtype, device=self.device)

            for sample_idx in range(n_mask_samples):
                mask_sample, sample_pixel_count = self.sample_mask_pixels(mask_prep_data, n_trim_pixels_tensor, sample_idx)
                pert_sample = mask_sample * pert
                output, loss = self.test_pert(x, y, pert_sample)
                pixel_sample_count += mask_sample
                pixel_loss_sum += mask_sample * (loss / sample_pixel_count).unsqueeze(1).unsqueeze(2).unsqueeze(3)

            pixel_sample_count.clamp_(min=1)  # Avoid zero values in tensor
            pixels_crit = pixel_loss_sum / pixel_sample_count

            best_crit_mask_indices = pixels_crit.view(self.batch_size, -1).topk(n_trim_pixels_tensor, dim=1, sorted=False)[
                1]
            best_crit_mask = self.compute_mask_from_ind(best_crit_mask_indices)
            best_crit_pert = best_crit_mask * pert
        return best_crit_pert, best_crit_mask

    def trim_pert_pixels(self, x, y, pert, mask, n_trim_pixels, trim_ratio,
                         mask_prep, mask_sample, n_mask_samples, mask_compute):
        with torch.no_grad():
            n_trim_pixels_tensor = torch.tensor(n_trim_pixels, dtype=torch.int, device=self.device)
            pixels_prob = trim_ratio * mask.view(self.batch_size, -1).float()
            mask_prep_data = mask_prep(pixels_prob, n_trim_pixels_tensor)
            self.mask_sample = mask_sample
            return mask_compute(x, y, pert, n_trim_pixels_tensor, mask_prep_data, n_mask_samples)

    def perturb_no_trim(self, x, y, mask, dpo_mean, dpo_std,
                        best_pert, best_loss, best_succ, pert_init, eval_pert=True):
        with torch.no_grad():
            loss, succ = self.eval_pert(x, y, pert_init)
            self.update_best(best_loss, loss,
                             [best_pert, best_succ],
                             [pert_init, succ])

            if self.report_info:
                all_best_succ = torch.zeros(self.n_iter + 1, self.batch_size, dtype=torch.bool, device=self.device)
                all_best_loss = torch.zeros(self.n_iter + 1, self.batch_size, dtype=self.dtype, device=self.device)
                all_best_succ[0] = best_succ
                all_best_loss[0] = best_loss

        self.set_dpo(dpo_mean, dpo_std)
        self.model.eval()
        pert = pert_init.clone().detach()
        for k in range(1, self.n_iter + 1):
            pert.requires_grad_()
            train_loss = self.criterion(self.model.forward(x + self.dpo(pert)), y)
            grad = torch.autograd.grad(train_loss.mean(), [pert])[0].detach()

            with torch.no_grad():
                pert = self.step(pert, grad, mask)
                eval_loss, succ = self.eval_pert(x, y, pert)
                self.update_best(best_loss, eval_loss,
                                 [best_pert, best_succ],
                                 [pert, succ])
                if self.report_info:
                    all_best_succ[k] = best_succ | all_best_succ[k - 1]
                    all_best_loss[k] = best_loss

        if self.report_info:
            return all_best_succ, all_best_loss
        return None, None

    def perturb(self, x, y, targeted=False):
        with torch.no_grad():
            self.set_params(x, targeted)
            self.clean_loss, self.clean_succ = self.eval_pert(x, y, pert=torch.zeros_like(x))
            best_l0_perts = torch.zeros_like(x).unsqueeze(0).repeat(self.n_l0_norms, 1, 1, 1, 1)
            best_l0_loss = self.clean_loss.clone().detach().unsqueeze(0).repeat(self.n_l0_norms, 1)
            best_l0_succ = self.clean_succ.clone().detach().unsqueeze(0).repeat(self.n_l0_norms, 1)

            if self.report_info:
                all_best_succ = torch.zeros(self.n_l0_norms, self.n_restarts, self.n_iter + 1, self.batch_size, dtype=torch.bool, device=self.device)
                all_best_loss = torch.zeros(self.n_l0_norms, self.n_restarts, self.n_iter + 1, self.batch_size, dtype=self.dtype, device=self.device)
                best_l0_norms = torch.zeros(self.n_l0_norms, self.batch_size, dtype=self.dtype, device=self.device)

        for rest in range(self.n_restarts):
            with torch.no_grad():
                mask = self.mask_ones_flat.clone().detach().view(self.mask_shape)
                trim_steps = self.restarts_trim_steps[rest]
                l0_indices = self.restarts_l0_indices[rest]
                l0_copy_indices = self.restarts_l0_copy_indices[rest]
                trim_steps_ratios = self.restarts_trim_steps_ratios[rest]
                dpo_mean_steps = self.restarts_dpo_mean_steps[rest]
                dpo_std_steps = self.restarts_dpo_std_steps[rest]
                steps_mask_prep = self.restarts_mask_prep[rest]
                steps_mask_sample = self.restarts_mask_sample[rest]
                steps_n_mask_samples = self.restarts_n_mask_samples[rest]
                steps_mask_compute = self.restarts_mask_compute[rest]

                if self.rand_init:
                    pert_init = self.random_initialization()
                    pert_init = self.project(pert_init, mask)
                else:
                    pert_init = torch.zeros_like(x)

            for trim_idx, n_trim_pixels in enumerate(trim_steps):
                with torch.no_grad():
                    l0_idx = l0_indices[trim_idx]
                    best_pert = best_l0_perts[l0_idx]
                    best_loss = best_l0_loss[l0_idx]
                    best_succ = best_l0_succ[l0_idx]
                    trim_ratio = trim_steps_ratios[trim_idx]
                    dpo_mean = dpo_mean_steps[trim_idx]
                    dpo_std = dpo_std_steps[trim_idx]
                    mask_prep = steps_mask_prep[trim_idx]
                    mask_sample = steps_mask_sample[trim_idx]
                    n_mask_samples = steps_n_mask_samples[trim_idx]
                    mask_compute = steps_mask_compute[trim_idx]

                no_trim_all_best_succ, no_trim_all_best_loss\
                    = self.perturb_no_trim(x, y, mask, dpo_mean, dpo_std,
                                           best_pert, best_loss, best_succ, pert_init)
                with torch.no_grad():
                    if self.report_info:
                        all_best_succ[l0_idx, rest] = no_trim_all_best_succ
                        all_best_loss[l0_idx, rest] = no_trim_all_best_loss
                        pert_l0 = best_pert.abs().view(self.batch_size, self.data_channels, -1).sum(dim=1).count_nonzero(1)
                        best_l0_norms[l0_idx] = pert_l0
                    if self.verbose:
                        print("Finished optimizing sparse perturbation on predetermined pixels")
                        print('max L0 in perturbation: ' + str(pert_l0.max().item()))
                        pert_l_inf = (best_pert.abs() / self.data_RGB_size).view(self.batch_size, -1).max(1)[0]
                        print('max L_inf in perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                            pert_l_inf.max(), (best_pert != best_pert).sum(), best_pert.max(), best_pert.min()))

                    pert_init, mask = \
                        self.trim_pert_pixels(x, y, best_pert, mask, n_trim_pixels, trim_ratio,
                                              mask_prep, mask_sample, n_mask_samples, mask_compute)

            with torch.no_grad():
                l0_idx = l0_indices[-1]
                best_pert = best_l0_perts[l0_idx]
                best_loss = best_l0_loss[l0_idx]
                best_succ = best_l0_succ[l0_idx]
            no_trim_all_best_succ, no_trim_all_best_loss \
                = self.perturb_no_trim(x, y, mask, self.post_trim_dpo_mean, self.post_trim_dpo_std,
                                       best_pert, best_loss, best_succ, pert_init)
            with torch.no_grad():
                if self.report_info:
                    all_best_succ[l0_idx, rest] = no_trim_all_best_succ
                    all_best_loss[l0_idx, rest] = no_trim_all_best_loss
                    pert_l0 = best_pert.abs().view(self.batch_size, self.data_channels, -1).sum(dim=1).count_nonzero(1)
                    best_l0_norms[l0_idx] = pert_l0
                    if len(l0_copy_indices):
                        all_best_succ[l0_copy_indices, rest] = all_best_succ[l0_copy_indices, rest - 1, -1].unsqueeze(1)
                        all_best_loss[l0_copy_indices, rest] = all_best_loss[l0_copy_indices, rest - 1, -1].unsqueeze(1)

                if self.verbose:
                    print("Finished optimizing perturbation without pixel trimming")
                    print('max L0 in perturbation: ' + str(pert_l0.max().item()))
                    pert_l_inf = (best_pert.abs() / self.data_RGB_size).view(self.batch_size, -1).max(1)[0]
                    print('max L_inf in perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                        pert_l_inf.max(), (best_pert != best_pert).sum(), best_pert.max(),
                        best_pert.min()))

        if self.report_info:
            return best_l0_perts, best_l0_norms, all_best_succ, all_best_loss
        l0_adv_pert = best_l0_perts[-1].clone().detach().unsqueeze(0)
        pert_l0_norm = l0_adv_pert.abs().view(self.batch_size, self.data_channels, -1).sum(dim=1).count_nonzero(
            1).unsqueeze(0)

        return l0_adv_pert, pert_l0_norm, None, None


