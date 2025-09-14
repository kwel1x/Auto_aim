# YOLO-based Real-time AI Assistant Core

这是一个基于 DXGI、TensorRT 和驱动级输入实现的实时 AI 视觉项目核心代码。

---

### 核心思路

1.  **截图**: 使用 DXGI Desktop Duplication 实现低延迟画面捕获。
2.  **推理**: 使用 ONNX Runtime + TensorRT 后端运行端到端 YOLO 模型。
3.  **输入**: 调用 `IbInputSimulator` 库，通过硬件驱动模拟鼠标移动。

---

### **⚠️ 重要声明 (Disclaimer)**

*   **本项目99%的代码由AI辅助生成**，因此所有代码都堆砌在 `main.cpp` 中，结构较为混乱，注释可能也不明确。本仓库旨在分享一种可行的技术思路，而非一个工程化的项目。
*   **仅上传了核心代码**，所有依赖环境（如 OpenCV, ONNX Runtime, CUDA 等）均需使用者**自行配置**。
*   仓库中的 `.onnx` 模型仅为示例，你需要**自行训练并替换**成你自己的模型。
*   **本项目仅供学习和技术交流使用**，严禁用于任何违规用途。一切后果由使用者自行承担。

---

### **依赖与致谢 (Dependencies & Credits)**

本项目能够实现，离不开以下优秀的开源项目：

*   **[IbInputSimulator](https://github.com/Chaoses-Ib/IbInputSimulator)** by Chaoses-Ib: 用于实现驱动级的鼠标输入模拟，是整个项目的关键一环。

---

### **运行**

1.  自行配置好所有依赖环境。
2.  将所有文件放入一个 C++ 项目中进行编译。
3.  确保 `.onnx` 模型和 `.dll` 文件在 `.exe` 旁边。
4.  **以管理员权限运行。**
