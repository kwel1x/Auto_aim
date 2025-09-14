#pragma once // 防止头文件被重复包含

#include <windows.h>
#include <cstdint> // 包含这个头文件以使用 uint32_t

// 声明我们将要使用的枚举
// 注意：根据您的新代码，我已将枚举放在了 Send 命名空间下
namespace Send {
    enum class Error {
        Success = 0, InvalidArgument = 1, DeviceNotFound = 2, DriverError = 3
    };
    enum class SendType {
        SendInput = 0, Logitech = 1, Razer = 2, DD = 3,
        MouClassInputInjection = 4, LogitechGHubNew = 5, AnyDriver = 100
    };
    enum class MoveMode {
        Absolute = 0, Relative = 1
    };
    enum class InitFlags {
        // ... 如果有的话
    };
}


// 使用 class 来封装所有与 DLL 相关的状态和行为
class InputSimulator {
public:
    InputSimulator();
    ~InputSimulator();

    Send::Error Init(Send::SendType type);
    // MoveRelative 的参数现在匹配新的函数
    void MoveRelative(int dx, int dy);
    // MoveAbsolute 暂时无法实现，因为我们不知道新函数如何处理绝对移动，先注释掉
    // void MoveAbsolute(int x, int y);

private:
    HMODULE hDll;

    // --- 使用新的、正确的函数指针类型 ---
    typedef Send::Error(__stdcall* PFN_IbSendInit)(Send::SendType type, Send::InitFlags flags, void* argument);
    typedef void(__stdcall* PFN_IbSendDestroy)();
    // 这是关键：定义新的鼠标移动函数指针
    typedef bool(__stdcall* PFN_IbSendMouseMove)(uint32_t dx, uint32_t dy, Send::MoveMode mode);

    // --- 更新函数指针变量 ---
    PFN_IbSendInit pIbSendInit;
    PFN_IbSendDestroy pIbSendDestroy;
    // 使用新的鼠标移动指针
    PFN_IbSendMouseMove pIbSendMouseMove;

    bool is_initialized_successfully;
};