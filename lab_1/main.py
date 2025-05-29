import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from math import sin, cos, pi, exp
from PIL import Image, ImageTk, ImageDraw

class SurfaceVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D")
        
        self.canvas = tk.Canvas(root, width=600, height=500, bg='white')
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        self.surface_var = tk.StringVar()
        self.surface_var.set("1")
        surfaces = [
            ("1. Спиральная поверхность", "1"),
            ("2. Мёбиус", "2"),
            ("3. Тор", "3"),
            ("4. Винтовая", "4"),
            ("5. Морская ракушка", "5")
        ]
        
        tk.Label(self.control_frame, text="Выбери своего бойца:").pack(anchor=tk.W)
        for text, value in surfaces:
            tk.Radiobutton(self.control_frame, text=text, variable=self.surface_var,
                          value=value, command=self.update_controls).pack(anchor=tk.W)
        
        self.param_frame = tk.Frame(self.control_frame)
        self.param_frame.pack(fill=tk.X, pady=10)
        
        self.grid_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.control_frame, text="Координатная сетка", variable=self.grid_var).pack(anchor=tk.W)
        
        tk.Label(self.control_frame, text="Поворот:").pack(anchor=tk.W)
        self.rotation_frame = tk.Frame(self.control_frame)
        self.rotation_frame.pack(fill=tk.X)
        
        tk.Label(self.rotation_frame, text="X:").pack(side=tk.LEFT)
        self.rot_x = tk.Scale(self.rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.rot_x.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(self.rotation_frame, text="Y:").pack(side=tk.LEFT)
        self.rot_y = tk.Scale(self.rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.rot_y.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(self.rotation_frame, text="Z:").pack(side=tk.LEFT)
        self.rot_z = tk.Scale(self.rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.rot_z.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.draw_button = tk.Button(self.control_frame, text="Айдерей санстрайк", command=self.draw_surface)
        self.draw_button.pack(fill=tk.X, pady=10)
        
        self.parameters = {}
        self.update_controls()
        
        self.draw_surface()
    
    def update_controls(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        surface_id = self.surface_var.get()
        self.parameters = {}
        
        if surface_id == "1":
            tk.Label(self.param_frame, text="Спиральная").pack(anchor=tk.W)
            self.add_parameter("alpha", 3.0, 1.0, 5.0, 0.2)
            self.add_parameter("beta", 0.8, 0.2, 1.5, 0.1)
            self.u_range = (0, 4*pi)
            self.v_range = (0, 2*pi)
            
        elif surface_id == "2": 
            tk.Label(self.param_frame, text="Мёбиус").pack(anchor=tk.W)
            self.add_parameter("alpha", 2.0, 1.0, 3.0, 0.2)
            self.add_parameter("beta", 1.0, 0.5, 2.0, 0.1)
            self.u_range = (0, 2*pi)
            self.v_range = (-0.5, 0.5)
            
        elif surface_id == "3":
            tk.Label(self.param_frame, text="Тор").pack(anchor=tk.W)
            self.add_parameter("alpha", 3.0, 2.0, 5.0, 0.3)
            self.add_parameter("beta", 1.0, 0.5, 2.0, 0.1)
            self.u_range = (0, 2*pi)
            self.v_range = (0, 2*pi)
            
        elif surface_id == "4": 
            tk.Label(self.param_frame, text="Винтовая").pack(anchor=tk.W)
            self.add_parameter("alpha", 1.0, 0.5, 2.0, 0.1)
            self.add_parameter("beta", 1.0, 0.5, 2.0, 0.1)
            self.u_range = (0, 4*pi)
            self.v_range = (-2, 2)
            
        elif surface_id == "5": 
            tk.Label(self.param_frame, text="Морская ракушка").pack(anchor=tk.W)
            self.add_parameter("alpha", 0.3, 0.1, 0.5, 0.05)
            self.add_parameter("beta", 0.1, 0.05, 0.2, 0.01)
            self.u_range = (0, 2*pi)
            self.v_range = (0, 6*pi)
    
    def add_parameter(self, name, default, min_val, max_val, step):
        frame = tk.Frame(self.param_frame)
        frame.pack(fill=tk.X, pady=2)
        
        tk.Label(frame, text=f"{name}:").pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=default)
        self.parameters[name] = var
        
        scale = tk.Scale(frame, from_=min_val, to=max_val, resolution=step,
                        orient=tk.HORIZONTAL, variable=var)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def draw_surface(self):
        surface_id = self.surface_var.get()
        
        params = {name: var.get() for name, var in self.parameters.items()}
        
        u_res = 50
        v_res = 30
        
        u_vals = np.linspace(self.u_range[0], self.u_range[1], u_res)
        v_vals = np.linspace(self.v_range[0], self.v_range[1], v_res)
        
        points = []
        
        for u in u_vals:
            row = []
            for v in v_vals:
                if surface_id == "1": 
                    x = (params['alpha'] + params['beta'] * cos(v)) * cos(u)
                    y = (params['alpha'] + params['beta'] * cos(v)) * sin(u)
                    z = params['beta'] * sin(v) + params['alpha'] * u
                elif surface_id == "2":
                    x = (params['alpha'] + v * cos(u/2)) * cos(u)
                    y = (params['alpha'] + v * cos(u/2)) * sin(u)
                    z = params['beta'] * v * sin(u/2)
                elif surface_id == "3":  
                    x = (params['alpha'] + params['beta'] * cos(v)) * cos(u)
                    y = (params['alpha'] + params['beta'] * cos(v)) * sin(u)
                    z = params['beta'] * sin(v)
                elif surface_id == "4": 
                    x = params['alpha'] * u * cos(u)
                    y = params['beta'] * u * sin(u)
                    z = v
                elif surface_id == "5": 
                    x = params['alpha'] * exp(params['beta'] * v) * cos(v) * (1 + cos(u))
                    y = params['alpha'] * exp(params['beta'] * v) * sin(v) * (1 + cos(u))
                    z = params['alpha'] * exp(params['beta'] * v) * sin(u)
                
                row.append((x, y, z))
            points.append(row)
        
        points = np.array(points)
        
        rot_x = np.radians(self.rot_x.get())
        rot_y = np.radians(self.rot_y.get())
        rot_z = np.radians(self.rot_z.get())
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x)],
            [0, np.sin(rot_x), np.cos(rot_x)]
        ])
        
        Ry = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        
        Rz = np.array([
            [np.cos(rot_z), -np.sin(rot_z), 0],
            [np.sin(rot_z), np.cos(rot_z), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        rotated_points = np.dot(points.reshape(-1, 3), R.T).reshape(points.shape)
        
        projected = rotated_points[:, :, :2]  
        
        grid_points = []
        if self.grid_var.get():
            x_axis = np.array([[x, 0, 0] for x in np.linspace(-5, 5, 20)])
            y_axis = np.array([[0, y, 0] for y in np.linspace(-5, 5, 20)])
            z_axis = np.array([[0, 0, z] for z in np.linspace(-5, 5, 20)])
            
            xy_grid = []
            for x in np.linspace(-5, 5, 10):
                for y in np.linspace(-5, 5, 10):
                    xy_grid.append([x, y, 0])
            
            xz_grid = []
            for x in np.linspace(-5, 5, 10):
                for z in np.linspace(-5, 5, 10):
                    xz_grid.append([x, 0, z])
            
            yz_grid = []
            for y in np.linspace(-5, 5, 10):
                for z in np.linspace(-5, 5, 10):
                    yz_grid.append([0, y, z])
            
            grid_points = {
                'x_axis': np.dot(x_axis, R.T),
                'y_axis': np.dot(y_axis, R.T),
                'z_axis': np.dot(z_axis, R.T),
                'xy_grid': np.dot(np.array(xy_grid), R.T),
                'xz_grid': np.dot(np.array(xz_grid), R.T),
                'yz_grid': np.dot(np.array(yz_grid), R.T)
            }
        
        all_points = projected.reshape(-1, 2)
        if grid_points:
            all_grid_points = np.concatenate([
                grid_points['x_axis'][:, :2],
                grid_points['y_axis'][:, :2],
                grid_points['z_axis'][:, :2],
                grid_points['xy_grid'][:, :2],
                grid_points['xz_grid'][:, :2],
                grid_points['yz_grid'][:, :2]
            ])
            all_points = np.concatenate([all_points, all_grid_points])
        
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        
        scale = 0.9 * min(500 / (max_vals[0] - min_vals[0]), 500 / (max_vals[1] - min_vals[1]))
        offset_x = 300 - scale * (max_vals[0] + min_vals[0]) / 2
        offset_y = 250 - scale * (max_vals[1] + min_vals[1]) / 2
        
        projected = scale * projected + np.array([offset_x, offset_y])
        
        if grid_points:
            scaled_grid = {}
            for key, points in grid_points.items():
                scaled_points = scale * points[:, :2] + np.array([offset_x, offset_y])
                scaled_grid[key] = scaled_points
            grid_points = scaled_grid
        
        img = Image.new('RGB', (600, 500), 'white')
        draw = ImageDraw.Draw(img)
        
        if self.grid_var.get() and grid_points:
            xy_points = grid_points['xy_grid']
            for i in range(0, len(xy_points), 10):
                for j in range(10):
                    if i+j+1 < len(xy_points):
                        draw.line([tuple(xy_points[i+j]), tuple(xy_points[i+j+1])], fill=(200, 200, 200), width=1)
            
            xz_points = grid_points['xz_grid']
            for i in range(0, len(xz_points), 10):
                for j in range(10):
                    if i+j+1 < len(xz_points):
                        draw.line([tuple(xz_points[i+j]), tuple(xz_points[i+j+1])], fill=(220, 220, 220), width=1)
            
            yz_points = grid_points['yz_grid']
            for i in range(0, len(yz_points), 10):
                for j in range(10):
                    if i+j+1 < len(yz_points):
                        draw.line([tuple(yz_points[i+j]), tuple(yz_points[i+j+1])], fill=(230, 230, 230), width=1)
            
            x_axis = grid_points['x_axis']
            y_axis = grid_points['y_axis']
            z_axis = grid_points['z_axis']
            
            for i in range(len(x_axis)-1):
                draw.line([tuple(x_axis[i]), tuple(x_axis[i+1])], fill=(255, 0, 0), width=2)
            
            for i in range(len(y_axis)-1):
                draw.line([tuple(y_axis[i]), tuple(y_axis[i+1])], fill=(0, 255, 0), width=2)
            
            for i in range(len(z_axis)-1):
                draw.line([tuple(z_axis[i]), tuple(z_axis[i+1])], fill=(0, 0, 255), width=2)
            
            origin = scale * np.array([0, 0]) + np.array([offset_x, offset_y])
            draw.ellipse([(origin[0]-3, origin[1]-3), (origin[0]+3, origin[1]+3)], fill=(0, 0, 0))
        
        for i in range(u_res - 1):
            for j in range(v_res - 1):
                p1 = projected[i, j]
                p2 = projected[i+1, j]
                p3 = projected[i+1, j+1]
                p4 = projected[i, j+1]
                
                avg_z = (rotated_points[i, j, 2] + rotated_points[i+1, j, 2] + 
                         rotated_points[i+1, j+1, 2] + rotated_points[i, j+1, 2]) / 4
                
                shade = int(255 * (1 - (avg_z + 2) / 4))
                color = (shade, shade, 255)
                
                draw.polygon([tuple(p1), tuple(p2), tuple(p3), tuple(p4)], fill=color, outline='black')
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = SurfaceVisualizer(root)
    root.mainloop()