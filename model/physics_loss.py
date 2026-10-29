import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, grid_spacing=1.0, enabled=None, var_mean=None, var_std=None):
        super().__init__()
        self.dx = grid_spacing
        self.dy = grid_spacing
        self.eps = 1e-6
        self.enabled = enabled

        if var_mean is not None and var_std is not None:
            self.register_buffer('var_mean', torch.tensor(var_mean, dtype=torch.float32).view(1, 8, 1, 1))
            self.register_buffer('var_std', torch.tensor(var_std, dtype=torch.float32).view(1, 8, 1, 1))
        else:
            self.var_mean = None
            self.var_std = None

    def _gradient_x(self, field):
        return (field[:, :, 2:] - field[:, :, :-2]) / (2 * self.dx + self.eps)

    def _gradient_y(self, field):
        return (field[:, 2:, :] - field[:, :-2, :]) / (2 * self.dy + self.eps)

    def _safe_mean(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.clamp(x.mean(), -1e6, 1e6)

    def _is_active(self, key):
        return self.enabled is None or key in self.enabled

    def forward(self, pred_fields):
        # 🔧 核心修复：将归一化输出还原为真实物理量（摄氏度、hPa、m/s 等）
        if self.var_mean is not None:
            pred_fields = pred_fields * self.var_std + self.var_mean

        tmax = pred_fields[:, 0, :, :]
        tmin = pred_fields[:, 1, :, :]
        prec = pred_fields[:, 2, :, :]
        wind = pred_fields[:, 3, :, :]
        pres = pred_fields[:, 4, :, :] / 100.0  # 🔧 核心修复：原始数据单位是 Pa，转为 hPa
        srad = pred_fields[:, 5, :, :]
        lrad = pred_fields[:, 6, :, :]
        shum = pred_fields[:, 7, :, :]

        tmax = torch.clamp(tmax, -50.0, 50.0)
        tmin = torch.clamp(tmin, -50.0, 50.0)
        prec = torch.clamp(prec, 0.0, 500.0)
        wind = torch.clamp(wind, 0.0, 50.0)
        pres = torch.clamp(pres, 300.0, 1100.0) # 🔧 修正范围
        shum = torch.clamp(shum, 0.0, 0.1)

        du_dx = self._gradient_x(wind)
        dv_dy = self._gradient_y(wind)
        dP_dx = self._gradient_x(pres)
        dP_dy = self._gradient_y(pres)

        tmax_crop = tmax[:, 1:-1, 1:-1]
        tmin_crop = tmin[:, 1:-1, 1:-1]
        prec_crop = prec[:, 1:-1, 1:-1]
        wind_crop = wind[:, 1:-1, 1:-1]
        pres_crop = pres[:, 1:-1, 1:-1]
        srad_crop = srad[:, 1:-1, 1:-1]
        lrad_crop = lrad[:, 1:-1, 1:-1]
        shum_crop = shum[:, 1:-1, 1:-1]

        du_dx_crop = du_dx[:, 1:-1, :]
        dv_dy_crop = dv_dy[:, :, 1:-1]
        dP_dx_crop = dP_dx[:, 1:-1, :]
        dP_dy_crop = dP_dy[:, :, 1:-1]

        device = pred_fields.device
        zero = torch.tensor(0.0, device=device)

        if self._is_active('cont'):
            L_cont = self._safe_mean((du_dx_crop + dv_dy_crop) ** 2)
        else:
            L_cont = zero

        T_avg_crop = (tmax_crop + tmin_crop) / 2 + 273.15
        T_avg_crop = torch.clamp(T_avg_crop, 200.0, 350.0)
        if self._is_active('thermo'):
            P_approx = 1013.25 * (T_avg_crop / 288.15) ** 5.256
            L_thermo = self._safe_mean((pres_crop - P_approx) ** 2)
        else:
            L_thermo = zero

        if self._is_active('mom'):
            L_mom = self._safe_mean((wind_crop + dP_dy_crop) ** 2 + (wind_crop - dP_dx_crop) ** 2)
        else:
            L_mom = zero

        if self._is_active('energy'):
            net_rad = (srad_crop - lrad_crop) * 0.1
            T_deviation = T_avg_crop - 273.15
            L_energy = self._safe_mean((T_deviation - net_rad) ** 2)
        else:
            L_energy = zero

        if self._is_active('phase'):
            phase_factor = torch.sigmoid((T_avg_crop - 273.15) / 5.0)
            L_phase = self._safe_mean((prec_crop * phase_factor) ** 2)
        else:
            L_phase = zero

        if self._is_active('covar'):
            T_std = T_avg_crop.std() + self.eps
            q_std = shum_crop.std() + self.eps
            T_norm = (T_avg_crop - T_avg_crop.mean()) / T_std
            q_norm = (shum_crop - shum_crop.mean()) / q_std
            L_covar = self._safe_mean((T_norm - q_norm) ** 2)
        else:
            L_covar = zero

        return {'cont': L_cont, 'thermo': L_thermo, 'mom': L_mom, 'energy': L_energy, 'phase': L_phase, 'covar': L_covar}