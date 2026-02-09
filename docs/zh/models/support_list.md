# 模型清单
DrivingSDK已支持下列开源模型，覆盖自动驾驶感知、规控、端到端领域典型模型及部分主流VLA、世界模型。列表中Released为Y的表示已通过版本测试发布，N的表示开发自验通过但未经过版本测试。

<table align=left>
    <tr>
        <td align=left>类别</td>
        <td align=left>模型</td>
        <td align=left>Atlas 800T A2性能（FPS）</td>
        <td align=left>竞品性能（FPS）</td>
        <td align=left>Released</td>
    </tr>
    <tr>
        <td rowspan=11>视觉感知</td>
        <td align=left><a href=../../../model_examples/CenterNet/README.md>CenterNet</a></td>
        <td align=left>1257.44</td>
        <td align=left>542</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/FCOS/README.md>FCOS-resnet</a></td>
        <td align=left>196</td>
        <td align=left>196</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/DETR/README.md>DETR</a></td>
        <td align=left>122</td>
        <td align=left>126</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Deformable-DETR/README.md>Deformable-DETR</a></td>
        <td align=left>63</td>
        <td align=left>65</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Mask2Former/README.md>Mask2Former</a></td>
        <td align=left>26.03</td>
        <td align=left>28.42</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/FCOS3D/README.md>FCOS3D</a></td>
        <td align=left>44.31</td>
        <td align=left>44.30</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Sparse4D/README.md>Sparse4D</a></td>
        <td align=left>70.59</td>
        <td align=left>65.75</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/DETR3D/README.md>DETR3D</a></td>
        <td align=left>14.35</td>
        <td align=left>14.28</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/StreamPETR/README.md>StreamPETR</a></td>
        <td align=left>26.016</td>
        <td align=left>25.397</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Yolov8/README.md>Yolov8</a></td>
        <td align=left>214.64</td>
        <td align=left>479.73</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/DinoV3/README.md>DinoV3</a></td>
        <td align=left>393.8</td>
        <td align=left>616.8</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td rowspan=6>点云感知</td>
        <td align=left><a href=../../../model_examples/PointPillar/README.md>PointPillar</a></td>
        <td align=left>70.79</td>
        <td align=left>60.75</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/CenterPoint/README.md>CenterPoint(2D)</a></td>
        <td align=left>66.16</td>
        <td align=left>85.712</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/CenterPoint/README.md>CenterPoint(3D)</a></td>
        <td align=left>39.41</td>
        <td align=left>48.48</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/SalsaNext/README.md>SalsaNext</a></td>
        <td align=left>197.2</td>
        <td align=left>241.6</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Panoptic-PolarNet/README.md>Panoptic-PolarNet</a></td>
        <td align=left>1.28</td>
        <td align=left>1.69</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/PointTransformerV3/README.md>PointTransformerV3</a></td>
        <td align=left>20.61</td>
        <td align=left>35.56</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td rowspan=7>BEV感知</td>
        <td align=left><a href=../../../model_examples/BEVFormer/README.md>BEVFormer</a></td>
        <td align=left>3.66</td>
        <td align=left>3.32</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/BEVDet/README.md>BEVDet</a></td>
        <td align=left>73.81</td>
        <td align=left>37.16</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/BEVDet4D/README.md>BEVDet4D</a></td>
        <td align=left>7.04</td>
        <td align=left>5.59</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/BEVDepth/README.md>BEVDepth</a></td>
        <td align=left>32.29</td>
        <td align=left>22.11</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/MatrixVT/README.md>MatrixVT</a></td>
        <td align=left>46.19</td>
        <td align=left>36.89</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td rowspan=2, align=left><a href=../../../model_examples/BEVNeXt/README.md>BEVNeXt</a></td>
        <td align=left>stage1: 16.568</td>
        <td align=left>stage1: 36.643</td>
        <td rowspan=2, align=left>N</td>
    </tr>
    <tr>
        <td align=left>stage2: 7.572</td>
        <td align=left>stage2: 11.651</td>
    </tr>
    <tr>
        <td rowspan=5>OCC感知</td>
        <td align=left><a href=../../../model_examples/PanoOcc/README.md>PanoOCC</a></td>
        <td align=left>4.32</td>
        <td align=left>4.87</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/FlashOCC/README.md>FlashOCC</a></td>
        <td align=left>104.85</td>
        <td align=left>67.98</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/FBOCC/README.md>FB-OCC</a></td>
        <td align=left>20.80</td>
        <td align=left>33.61</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/TPVFormer/README.md>TPVFormer</a></td>
        <td align=left>6.69</td>
        <td align=left>10.32</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/SurroundOcc/README.md>SurroundOCC</a></td>
        <td align=left>7.59</td>
        <td align=left>7.78</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td rowspan=1>融合感知</td>
        <td align=left><a href=../../../model_examples/BEVFusion/README.md>BEVFusion</a></td>
        <td align=left>26.46</td>
        <td align=left>22.54</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td rowspan=4>Lane&Map</td>
        <td align=left><a href=../../../model_examples/MapTR/README.md>MapTR</a></td>
        <td align=left>34.85</td>
        <td align=left>33.2</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/MapTRv2/README.md>MapTRv2</a></td>
        <td align=left>1257.44</td>
        <td align=left>542</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/PivotNet/README.md>PivotNet</a></td>
        <td align=left>23.03</td>
        <td align=left>21.91</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/LaneSegNet/README.md>LaneSegNet</a></td>
        <td align=left>18</td>
        <td align=left>23.75</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td rowspan=7>规控</td>
        <td align=left><a href=../../../model_examples/QCNet/README.md>QCNet</a></td>
        <td align=left>75.29</td>
        <td align=left>94.11</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/MultiPath++/README.md>MultiPath++</a></td>
        <td align=left>149.53</td>
        <td align=left>198.14</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/GameFormer/README.md>GameFormer</a></td>
        <td align=left>7501.8</td>
        <td align=left>6400</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/DenseTNT/README.md>DenseTNT</a></td>
        <td align=left>166</td>
        <td align=left>237</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/GameFormer-Planner/README.md>GameFormer-Planner</a></td>
        <td align=left>5319</td>
        <td align=left>5185</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Diffusion-Planner/README.md>Diffusion-Planner</a></td>
        <td align=left>5808.13</td>
        <td align=left>5304.32</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/HiVT/README.md>HiVT</a></td>
        <td align=left>645</td>
        <td align=left>652</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td rowspan=6>端到端</td>
        <td rowspan=2, align=left><a href=../../../model_examples/UniAD/README.md>UniAD</a></td>
        <td align=left>stage1: 1.002</td>
        <td align=left>stage1: 1.359</td>
        <td rowspan=2, align=left>Y</td>
    </tr>
    <tr>
        <td align=left>stage2: 1.554</td>
        <td align=left>stage2: 2</td>
    </tr>
    <tr>
        <td rowspan=2, align=left><a href=../../../model_examples/SparseDrive/README.md>SparseDrive</a></td>
        <td align=left>stage1: 46.3</td>
        <td align=left>stage1: 41.0</td>
        <td rowspan=2, align=left>Y</td>
    </tr>
    <tr>
        <td align=left>stage2: 37.9</td>
        <td align=left>stage2: 35.2</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/DiffusionDrive/README.md>DiffusionDrive</a></td>
        <td align=left>28.43</td>
        <td align=left>30.53</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/VAD/README.md>VAD</a></td>
        <td align=left>4.121</td>
        <td align=left>7.048</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td rowspan=3>LLM&VLM</td>
        <td align=left><a href=../../../model_examples/LMDrive/README.md>LMDrive</a></td>
        <td align=left>8.02</td>
        <td align=left>13.85</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Senna/README.md>Senna</a></td>
        <td align=left>1.376</td>
        <td align=left>1.824</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/HPTR/README.md>HPTR</a></td>
        <td align=left>25.12</td>
        <td align=left>36.07</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td rowspan=6>VLA</td>
        <td align=left><a href=../../../model_examples/OpenVLA/README.md>OpenVLA</a></td>
        <td align=left>56.14</td>
        <td align=left>73.12</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td rowspan=2, align=left><a href=../../../model_examples/Dexvla/README.md>Dexvla</a></td>
        <td align=left>stage2: 16.72</td>
        <td align=left>stage2: 18.88</td>
        <td rowspan=2, align=left>Y</td>
    </tr>
    <tr>
        <td align=left>stage3: 15.85</td>
        <td align=left>stage3: 18.67</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Pi-0/README.md>Pi-0</a></td>
        <td align=left>116.36</td>
        <td align=left>136.17</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Pi-0.5/README.md>Pi-0.5</a></td>
        <td align=left>2335(A3)</td>
        <td align=left>1115(竞品H)</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/GR00T-N1.5/README.md>GR00T-N1.5</a></td>
        <td align=left>337.35</td>
        <td align=left>276.38</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td rowspan=7>World Model</td>
        <td align=left><a href=../../../model_examples/OpenDWM/README.md>OpenDWM</a></td>
        <td align=left>1.82</td>
        <td align=left>1.82</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/VGGT/README.md>VGGT</a></td>
        <td align=left>25.04</td>
        <td align=left>15.30</td>
        <td align=left>Y</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Cosmos-Predict2/README.md>Cosmos-Predict2</a></td>
        <td align=left>-</td>
        <td align=left>-</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Cosmos-Transfer1/README.md>Cosmos-Transfer1</a></td>
        <td align=left>-</td>
        <td align=left>-</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/Cosmos-Drive-Dreams/README.md>Cosmos-Drive-Dreams</a></td>
        <td align=left>-</td>
        <td align=left>-</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/MagicDrive-V2/README.md>MagicDrive-V2</a></td>
        <td align=left>stage1: 0.83</td>
        <td align=left>stage1: 1.5</td>
        <td align=left>N</td>
    </tr>
    <tr>
        <td align=left><a href=../../../model_examples/NWM/README.md>Navigation World Models</a></td>
        <td align=left>363.39</td>
        <td align=left>383.06</td>
        <td align=left>N</td>
    </tr>
</table>

</br>