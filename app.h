#ifndef APP_H
#define APP_H

#include <iostream>
#include <set>
#include <map>
#include <unordered_set>
#include <optional>
#include <fstream>

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

class App {
	struct SwapchainSupportDetails {
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;

		bool isAdequate() { return !formats.empty() && !presentModes.empty(); }
	};

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
	};

	struct SwapchainData {
		vk::SwapchainKHR swapchain;
		vk::Extent2D extent;
		vk::Format format;

		std::vector<vk::ImageView> imageViews;
		std::vector<vk::Framebuffer> framebuffers;
	};

	public:
		App();
		~App();

		void run();

	private:
		// Window
		void createWindow();
		static void windowErrorCallback(int error_code, const char *description);

		// Instance & Sufrace
		void createInstance();
		bool extensionsSupported(const std::vector<const char *>& extensions);
		bool layersSupported();
		void createSurface();

		// Vulkan validation layer
		void setDebugMessenger();
		vk::DebugUtilsMessengerCreateInfoEXT getDebugMessengerInfo();
		static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanErrorCallback(
									VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
									VkDebugUtilsMessageTypeFlagsEXT messageType,
									const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
									void *pUserData);

		// Device
		void pickPhysicalDevice();
		int rateDevice(vk::PhysicalDevice physicalDevice);
		bool deviceExtensionsSupported(vk::PhysicalDevice physicalDevice);
		void createDevice();

		SwapchainSupportDetails querySwapchainSupport(vk::PhysicalDevice physicalDevice);
		QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice physicalDevice);

		// Swapchain
		void createSwapchain();
		void createImageViews();

		vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
				const std::vector<vk::SurfaceFormatKHR> &availableFormats);
		vk::PresentModeKHR chooseSwapPresentMode(
				const std::vector<vk::PresentModeKHR> &availablePresentModes);
		vk::Extent2D chooseSwapchainExtent(vk::SurfaceCapabilitiesKHR capabilities);

		// Pipeline
		void createRenderPass();
		void createPipelineLayout();
		void createFramebuffers();

		[[nodiscard]] std::vector<char> readFile(const std::string &filepath);
		[[nodiscard]] vk::ShaderModule createShaderModule(const std::vector<char> &code);
		void createPipeline();

		// Command buffer
		void createCommandPool();
		void createCommandBuffers();
		void recordCommandBuffer(vk::CommandBuffer commandBuffer, vk::Framebuffer framebuffer);

		// Rendering
		void createSyncObjects();
		void renderTriangle();

	private:
		// Window
		GLFWwindow *window;
		const unsigned int width{600};
		const unsigned int height{480};

		// Vulkan
		vk::Instance instance;
		vk::SurfaceKHR surface;

		// Validation
		vk::DispatchLoaderDynamic dldi;
		vk::DebugUtilsMessengerEXT debugMessenger;

		// Device
		vk::PhysicalDevice physicalDevice;
		vk::Device device;
		vk::Queue graphicsQueue;
		vk::Queue presentQueue;

		// Swapchain
		SwapchainData swapchainData;

		// Pipeline
		vk::PipelineLayout pipelineLayout;
		vk::RenderPass renderPass;
		vk::Pipeline pipeline;

		// Command buffer
		vk::CommandPool commandPool;
		std::vector<vk::CommandBuffer> commandBuffers;

		// Synchronization
		vk::Semaphore imageAvailableSemaphore;
		vk::Semaphore renderFinishedSemaphore;
		vk::Fence inFlightFence;
};

#endif // APP_H
