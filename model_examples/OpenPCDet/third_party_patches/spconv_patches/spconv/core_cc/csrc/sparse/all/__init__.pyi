from typing import overload, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from pccm.stubs import EnumValue, EnumClassValue
from cumm.tensorview import Tensor
class ThrustCustomAllocatorV2:
    alloc_func: Callable[int, int]
class SpconvOps:
    @staticmethod
    def generate_conv_inds_cpu(indices: Tensor, indice_pairs: Tensor, out_inds: Tensor, indice_num_per_loc: Tensor, batch_size: int, output_dims: List[int], input_dims: List[int], ksize: List[int], stride: List[int], padding: List[int], dilation: List[int], transposed: bool = False) -> int: 
        """
        Args:
            indices: 
            indice_pairs: 
            out_inds: 
            indice_num_per_loc: 
            batch_size: 
            output_dims: 
            input_dims: 
            ksize: 
            stride: 
            padding: 
            dilation: 
            transposed: 
        """
        ...
    @staticmethod
    def generate_subm_conv_inds_cpu(indices: Tensor, indice_pairs: Tensor, out_inds: Tensor, indice_num_per_loc: Tensor, batch_size: int, input_dims: List[int], ksize: List[int], dilation: List[int]) -> int: 
        """
        Args:
            indices: 
            indice_pairs: 
            out_inds: 
            indice_num_per_loc: 
            batch_size: 
            input_dims: 
            ksize: 
            dilation: 
        """
        ...
    @staticmethod
    def maxpool_forward_cpu(out: Tensor, inp: Tensor, out_inds: Tensor, in_inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            out_inds: 
            in_inds: 
        """
        ...
    @staticmethod
    def maxpool_backward_cpu(out: Tensor, inp: Tensor, dout: Tensor, dinp: Tensor, out_inds: Tensor, in_inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            dout: 
            dinp: 
            out_inds: 
            in_inds: 
        """
        ...
    @staticmethod
    def gather_cpu(out: Tensor, inp: Tensor, inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            inds: 
        """
        ...
    @staticmethod
    def scatter_add_cpu(out: Tensor, inp: Tensor, inds: Tensor) -> None: 
        """
        Args:
            out: 
            inp: 
            inds: 
        """
        ...
    @staticmethod
    def calc_point2voxel_meta_data(vsize_xyz: List[float], coors_range_xyz: List[float]) -> Tuple[List[float], List[int], List[int], List[float]]: 
        """
        Args:
            vsize_xyz: 
            coors_range_xyz: 
        """
        ...
    @staticmethod
    def point2voxel_cpu(points: Tensor, voxels: Tensor, indices: Tensor, num_per_voxel: Tensor, densehashdata: Tensor, pc_voxel_id: Tensor, vsize: List[float], grid_size: List[int], grid_stride: List[int], coors_range: List[float], empty_mean: bool = False, clear_voxels: bool = True) -> Tuple[Tensor, Tensor, Tensor]: 
        """
        Args:
            points: 
            voxels: 
            indices: 
            num_per_voxel: 
            densehashdata: 
            pc_voxel_id: 
            vsize: 
            grid_size: 
            grid_stride: 
            coors_range: 
            empty_mean: 
            clear_voxels: 
        """
        ...
