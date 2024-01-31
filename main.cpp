#include "app.h"

/**
 * Debug loggin
 */
template <typename T> void info(T param) {
	std::cerr << param << std::endl;
}

template <typename T, typename ...Args> void info(T param, Args ...args) {
	std::cerr << param;
	info(args...);
}

#ifdef NDEBUG
	const bool enabledDebug = false;
	#define INFO(...)
#else
	const bool enabledDebug = true;
	#define INFO(...) info(__VA_ARGS__)
#endif
	
/**
 * App initialization, destruction
 */
App::App() {
	createWindow();

	createInstance();
	setDebugMessenger();
	createSurface();

	pickPhysicalDevice();
	createDevice();

	createSwapchain();
	createImageViews();

	createRenderPass();
	createFramebuffers();
	createPipelineLayout();
	createPipeline();

	createCommandPool();
	createCommandBuffers();
	createSyncObjects();
}

App::~App() {
	device.waitIdle();

	device.destroyFence(inFlightFence);
	device.destroySemaphore(imageAvailableSemaphore);
	device.destroySemaphore(renderFinishedSemaphore);
	device.destroyCommandPool(commandPool);

	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyRenderPass(renderPass);

	for (size_t i = 0; i < swapchainData.imageViews.size(); i++) {
		device.destroyFramebuffer(swapchainData.framebuffers[i]);
		device.destroyImageView(swapchainData.imageViews[i]);
	}

	device.destroySwapchainKHR(swapchainData.swapchain);
	device.destroy();

	if (enabledDebug)
		instance.destroyDebugUtilsMessengerEXT(debugMessenger, nullptr, dldi);

	instance.destroySurfaceKHR(surface);
	instance.destroy();

	glfwTerminate();
}

/**
 * Initialize GLFW, set error callback
 */
void App::createWindow() {
	if (!glfwInit())
		throw std::runtime_error("Unable to initialize GLFW\n");

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);	// disable context (OpenGL)
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);		// disable resizing

	window = glfwCreateWindow(width, height, "Vulkan-Setup", nullptr, nullptr);
	if (window == nullptr) {
		glfwTerminate();
		throw std::runtime_error("Unable to create GLFWwindow\n");
	}

	glfwSetErrorCallback(windowErrorCallback);
}

void App::windowErrorCallback(int error_code, const char *description) {
	std::ostringstream errMessage;

	errMessage << "GLFW ERROR (" << error_code << "): " << description << std::endl;
	throw std::runtime_error(errMessage.str());
}

/**
 * Create instance & surface
 */
std::vector<const char *> layers {};
void App::createInstance() {
	uint32_t instanceVersion = vk::enumerateInstanceVersion();

	INFO("Vulkan version: ",
			 vk::apiVersionMajor(instanceVersion), ".",
			 vk::apiVersionMinor(instanceVersion), ".",
			 vk::apiVersionPatch(instanceVersion)); 

	instanceVersion &= ~(0xFFFU); // zero out patch number

	uint32_t extensionCount {0};
	const char **requiredExntensions = glfwGetRequiredInstanceExtensions(&extensionCount);
	std::vector<const char *> extensions {requiredExntensions, requiredExntensions + extensionCount};

	if (enabledDebug) {
		extensions.push_back("VK_EXT_debug_utils");
		layers.push_back("VK_LAYER_KHRONOS_validation");

	}

	vk::ApplicationInfo appInfo{};
	appInfo.pApplicationName = "Vulkan-Setup";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = instanceVersion;

	vk::InstanceCreateInfo instanceInfo{};
	instanceInfo.flags = vk::InstanceCreateFlags();
	instanceInfo.pApplicationInfo = &appInfo;
	instanceInfo.enabledLayerCount = layers.size();
	instanceInfo.ppEnabledLayerNames = layers.data();
	instanceInfo.enabledExtensionCount = extensions.size();
	instanceInfo.ppEnabledExtensionNames = extensions.data();

	if (enabledDebug) {
		auto debugMessengerInfo = getDebugMessengerInfo();
		instanceInfo.pNext = &debugMessengerInfo;
	}

	if (!extensionsSupported(extensions) && !layersSupported())
		throw std::runtime_error("Unable to create instance\n");

	instance = vk::createInstance(instanceInfo);
	dldi = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr);
}

bool App::extensionsSupported(const std::vector<const char *> &extensions) {
	std::unordered_set<std::string> requiredExtensions {
		extensions.begin(), extensions.end() };

	for (const auto &extension : vk::enumerateInstanceExtensionProperties())
		requiredExtensions.erase(extension.extensionName);

	if (requiredExtensions.empty())
		return true;

	for (const std::string &unsupportedExtension : requiredExtensions)
		INFO("Extension: ", unsupportedExtension, " is not supported");
	return false;
}

bool App::layersSupported() {
	std::unordered_set<std::string> requiredLayers {
		layers.begin(), layers.end() };

	for (const auto &layer: vk::enumerateInstanceLayerProperties())
		requiredLayers.erase(layer.layerName);

	if (requiredLayers.empty())
		return true;

	for (const std::string &unsupportedLayer : requiredLayers)
		INFO("Layer: ", unsupportedLayer, "is not supported");
	return false;
}

void App::createSurface() {
	VkSurfaceKHR c_surface;
	if (glfwCreateWindowSurface(instance, window, nullptr, &c_surface) != VK_SUCCESS)
		throw std::runtime_error("Unable to create window surface\n");

	surface = c_surface;
}

/**
 * Vulkan validation
 */
VKAPI_ATTR VkBool32 VKAPI_CALL App::vulkanErrorCallback(
							VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
							VkDebugUtilsMessageTypeFlagsEXT messageType,
							const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
							void *pUserData) {
	INFO("[VK_VALIDATION_LAYER] ", messageType, ": ", pCallbackData->pMessage);
	return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT App::getDebugMessengerInfo() {
	return vk::DebugUtilsMessengerCreateInfoEXT {
		vk::DebugUtilsMessengerCreateFlagsEXT(),
		// vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
		// vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
		vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
		vulkanErrorCallback
	};
}

void App::setDebugMessenger() {
	if (!enabledDebug) return;

	auto debugMessengerInfo = getDebugMessengerInfo();
	debugMessenger = instance.createDebugUtilsMessengerEXT(debugMessengerInfo, nullptr, dldi);
}

/**
 * Pick physical device
 */
std::vector<const char *> deviceExtensions {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

void App::pickPhysicalDevice() {
	std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
	std::multimap<int, vk::PhysicalDevice> candidates;
	
	if (devices.size() == 0)
		throw std::runtime_error("Failed to find GPU with Vulkan support\n");

	for (const auto &device : devices)
		candidates.insert(std::pair(rateDevice(device), device));

	if (candidates.rbegin()->first == 0)
		throw std::runtime_error("Failed to find suitable GPU\n");

	physicalDevice = candidates.rbegin()->second;
}

int App::rateDevice(vk::PhysicalDevice physicalDevice) {
	int score {1};
	vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();

	// favor discrete over integrated GPU
	if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
		score += 1000;

	if (!deviceExtensionsSupported(physicalDevice))
		return 0;

	auto swapchainSupport = querySwapchainSupport(physicalDevice);
	auto queueIndices = findQueueFamilies(physicalDevice);

	if (!swapchainSupport.isAdequate() || !queueIndices.isComplete())
		return 0;

	return score;
}

bool App::deviceExtensionsSupported(vk::PhysicalDevice physicalDevice) {
	std::unordered_set<std::string> requiredDeviceExtensions {
		deviceExtensions.begin(), deviceExtensions.end() };

	for (const auto &extension : physicalDevice.enumerateDeviceExtensionProperties())
		requiredDeviceExtensions.erase(extension.extensionName);

	if (requiredDeviceExtensions.empty())
		return true;

	for (const std::string &unsupportedExtension : requiredDeviceExtensions)
		INFO("Device extension: ", unsupportedExtension, " is not supported");
	return false;
}

App::SwapchainSupportDetails App::querySwapchainSupport(vk::PhysicalDevice physicalDevice) {
	SwapchainSupportDetails swapchainDetails;
	swapchainDetails.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	swapchainDetails.formats = physicalDevice.getSurfaceFormatsKHR(surface);
	swapchainDetails.presentModes = physicalDevice.getSurfacePresentModesKHR(surface);

	return swapchainDetails;
}

App::QueueFamilyIndices App::findQueueFamilies(vk::PhysicalDevice physicalDevice) {
	QueueFamilyIndices queueIndices;
	uint32_t i = 0;

	for (const auto &queueFamily : physicalDevice.getQueueFamilyProperties()) {
		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
			queueIndices.graphicsFamily = i;

		if (physicalDevice.getSurfaceSupportKHR(i, surface))
			queueIndices.presentFamily = i;

		if (queueIndices.isComplete())
			break;
		i++;
	}

	return queueIndices;
}

/**
 * Create logical device
 */
void App::createDevice() {
	auto queueIndices = findQueueFamilies(physicalDevice);

	std::vector<vk::DeviceQueueCreateInfo> queueInfos;
	std::set<uint32_t> uniqueQueueFamilies { 
		queueIndices.graphicsFamily.value(),
		queueIndices.presentFamily.value() };

	float queuePriority {1.0f};
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		vk::DeviceQueueCreateInfo queueInfo{};
		queueInfo.flags = vk::DeviceQueueCreateFlags();
		queueInfo.pQueuePriorities = &queuePriority;
		queueInfo.queueCount = 1;
		queueInfo.queueFamilyIndex = queueFamily;

		queueInfos.push_back(queueInfo);
	}

	vk::DeviceCreateInfo deviceInfo{};
	deviceInfo.flags = vk::DeviceCreateFlags();
	deviceInfo.queueCreateInfoCount = queueInfos.size();
	deviceInfo.pQueueCreateInfos = queueInfos.data();
	deviceInfo.enabledLayerCount = layers.size();
	deviceInfo.ppEnabledLayerNames = layers.data();
	deviceInfo.enabledExtensionCount = deviceExtensions.size();
	deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();

	device = physicalDevice.createDevice(deviceInfo);
	graphicsQueue = device.getQueue(queueIndices.graphicsFamily.value(), 0);
	presentQueue = device.getQueue(queueIndices.presentFamily.value(), 0);
}

/**
 * Create swapchain
 */
vk::SurfaceFormatKHR App::chooseSwapSurfaceFormat(
		const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
	for (const auto &format : availableFormats) {
		if (format.format == vk::Format::eB8G8R8A8Srgb &&
				format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return format;
		}
	}

	return availableFormats.front();
}

vk::PresentModeKHR App::chooseSwapPresentMode(
		const std::vector<vk::PresentModeKHR> &availablePresentModes) {
	for (const auto &presentMode : availablePresentModes) {
		if (presentMode == vk::PresentModeKHR::eMailbox) {
			return presentMode;
		}
	}

	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D App::chooseSwapchainExtent(vk::SurfaceCapabilitiesKHR capabilities) {
	if (capabilities.currentExtent.width != UINT32_MAX)
		return capabilities.currentExtent;

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	vk::Extent2D extent {
		std::clamp(static_cast<uint32_t>(width), 
							 capabilities.minImageExtent.width, 
							 capabilities.maxImageExtent.width), 
		std::clamp(static_cast<uint32_t>(height),
							 capabilities.minImageExtent.height, 
							 capabilities.maxImageExtent.height)
	};

	return extent;
}

void App::createSwapchain() {
	auto swapchainSupport = querySwapchainSupport(physicalDevice);
	auto indices = findQueueFamilies(physicalDevice);

	vk::SurfaceFormatKHR format = chooseSwapSurfaceFormat(swapchainSupport.formats);
	vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
	vk::Extent2D extent = chooseSwapchainExtent(swapchainSupport.capabilities);

	uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
	if (swapchainSupport.capabilities.maxImageCount != 0 &&
			imageCount > swapchainSupport.capabilities.maxImageCount) {
		imageCount = swapchainSupport.capabilities.maxImageCount;
	}

	vk::SwapchainCreateInfoKHR swapchainInfo{};
	swapchainInfo.flags = vk::SwapchainCreateFlagsKHR();
	swapchainInfo.surface = surface;
	swapchainInfo.minImageCount = imageCount;
	swapchainInfo.imageFormat = format.format;
	swapchainInfo.imageColorSpace = format.colorSpace;
	swapchainInfo.imageExtent = extent;
	swapchainInfo.imageArrayLayers = 1;
	swapchainInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

	if (indices.graphicsFamily != indices.presentFamily) {
		uint32_t queueFamilyIndices[] { 
			indices.graphicsFamily.value(),
			indices.presentFamily.value()
		};

		swapchainInfo.imageSharingMode = vk::SharingMode::eConcurrent;
		swapchainInfo.queueFamilyIndexCount = 2;
		swapchainInfo.pQueueFamilyIndices = queueFamilyIndices;
	} else {
		swapchainInfo.imageSharingMode = vk::SharingMode::eExclusive;
	}

	swapchainInfo.preTransform = swapchainSupport.capabilities.currentTransform;
	swapchainInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	swapchainInfo.presentMode = presentMode;
	swapchainInfo.clipped = VK_TRUE;

	swapchainData.swapchain = device.createSwapchainKHR(swapchainInfo);
	swapchainData.format = format.format;
	swapchainData.extent = extent;
}

void App::createImageViews() {
	std::vector<vk::Image> swapchainImages = device.getSwapchainImagesKHR(swapchainData.swapchain);

	for (const auto &image : swapchainImages) {
		vk::ImageViewCreateInfo imageViewInfo{};
		imageViewInfo.flags = vk::ImageViewCreateFlags();
		imageViewInfo.image = image;
		imageViewInfo.viewType = vk::ImageViewType::e2D;
		imageViewInfo.format = swapchainData.format;
		imageViewInfo.components = { 
			vk::ComponentSwizzle::eR,
			vk::ComponentSwizzle::eG,
			vk::ComponentSwizzle::eB,
			vk::ComponentSwizzle::eA 
		};
		imageViewInfo.subresourceRange = {
			vk::ImageAspectFlagBits::eColor,
			0, 1, 0, 1 
		};

		swapchainData.imageViews.push_back(device.createImageView(imageViewInfo));
	}
}

/**
 * Create pipeline
 */
void App::createRenderPass() {
	vk::AttachmentDescription attachment{};
	attachment.flags = vk::AttachmentDescriptionFlags();
	attachment.format = swapchainData.format;
	attachment.samples = vk::SampleCountFlagBits::e1;
	attachment.loadOp = vk::AttachmentLoadOp::eClear;
	attachment.storeOp = vk::AttachmentStoreOp::eStore;
	attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	attachment.initialLayout = vk::ImageLayout::eUndefined;
	attachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	vk::AttachmentReference attachmentRef{0, vk::ImageLayout::eColorAttachmentOptimal};

	// subpass
	vk::SubpassDescription subpass{};
	subpass.flags = vk::SubpassDescriptionFlags();
	subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &attachmentRef;

	// subpass dependencies
	vk::SubpassDependency dependency{};

	// renderPass
	vk::RenderPassCreateInfo renderPassInfo{};
	renderPassInfo.flags = vk::RenderPassCreateFlags();
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &attachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	renderPass = device.createRenderPass(renderPassInfo);
}

void App::createPipelineLayout() {
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.flags = vk::PipelineLayoutCreateFlags();
	pipelineLayoutInfo.setLayoutCount = 0;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
}

void App::createFramebuffers() {
	for (const auto &imageView : swapchainData.imageViews) {
		vk::FramebufferCreateInfo framebufferInfo{};
		framebufferInfo.flags = vk::FramebufferCreateFlags();
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = &imageView;
		framebufferInfo.width = swapchainData.extent.width;
		framebufferInfo.height = swapchainData.extent.height;
		framebufferInfo.layers = 1; 

		swapchainData.framebuffers.push_back(device.createFramebuffer(framebufferInfo));
	}
}

std::vector<char> App::readFile(const std::string &filepath) {
	std::ifstream file(filepath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		std::ostringstream error;
		std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);

		error << "Unable to load shader \"" << filename << "\"" << std::endl;
		throw std::runtime_error(error.str());
	}

	size_t filesize{static_cast<size_t>(file.tellg())};
	std::vector<char> buffer(filesize);

	file.seekg(0);
	file.read(buffer.data(), filesize);
	file.close();

	return buffer;
}

vk::ShaderModule App::createShaderModule(const std::vector<char> &code) {
	vk::ShaderModuleCreateInfo shaderModuleInfo{};
	shaderModuleInfo.flags = vk::ShaderModuleCreateFlags();
	shaderModuleInfo.codeSize = code.size();
	shaderModuleInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

	return device.createShaderModule(shaderModuleInfo);
}

const std::string vertexShaderPath {"../shaders/vertex.spv"};
const std::string fragmentShaderPath {"../shaders/fragment.spv"};

std::vector<vk::DynamicState> dynamicStates {
	vk::DynamicState::eViewport,
	vk::DynamicState::eScissor
};

void App::createPipeline() {
	// Shader stages
	auto vertexShaderCode = readFile(vertexShaderPath);
	auto fragmentShaderCode = readFile(fragmentShaderPath);

	vk::ShaderModule vertexModule = createShaderModule(vertexShaderCode);
	vk::ShaderModule fragmentModule = createShaderModule(fragmentShaderCode);

	vk::PipelineShaderStageCreateInfo vertexStage{};
	vertexStage.flags = vk::PipelineShaderStageCreateFlags();
	vertexStage.stage = vk::ShaderStageFlagBits::eVertex;
	vertexStage.module = vertexModule;
	vertexStage.pName = "main";

	vk::PipelineShaderStageCreateInfo fragmentStage{};
	fragmentStage.flags = vk::PipelineShaderStageCreateFlags();
	fragmentStage.stage = vk::ShaderStageFlagBits::eFragment;
	fragmentStage.module = fragmentModule;
	fragmentStage.pName = "main";

	vk::PipelineShaderStageCreateInfo shaderStageInfos[] { vertexStage, fragmentStage };

	// Vertex input
	vk::PipelineVertexInputStateCreateInfo vertexInputState{}; // no vertex buffer just yet
	vertexInputState.flags = vk::PipelineVertexInputStateCreateFlags();
	vertexInputState.vertexBindingDescriptionCount = 0;
	vertexInputState.vertexAttributeDescriptionCount = 0;

	// Input assembly
	vk::PipelineInputAssemblyStateCreateInfo assemblyState{};
	assemblyState.flags = vk::PipelineInputAssemblyStateCreateFlags();
	assemblyState.topology = vk::PrimitiveTopology::eTriangleList;
	assemblyState.primitiveRestartEnable = false;
	
	// Viewport
	vk::Viewport viewport;
	viewport.x = 0;
	viewport.y = 0;
	viewport.width = swapchainData.extent.width;
	viewport.height = swapchainData.extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	vk::Rect2D scissor;
	scissor.extent = swapchainData.extent;
	scissor.offset = vk::Offset2D(0, 0);

	vk::PipelineViewportStateCreateInfo viewportState{};
	viewportState.flags = vk::PipelineViewportStateCreateFlags();
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	// Dynamic
	vk::PipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.flags = vk::PipelineDynamicStateCreateFlags();
	dynamicState.dynamicStateCount = dynamicStates.size();
	dynamicState.pDynamicStates = dynamicStates.data();

	// Rasterization
	vk::PipelineRasterizationStateCreateInfo rasterizationState{};
	rasterizationState.flags = vk::PipelineRasterizationStateCreateFlags();
	rasterizationState.depthClampEnable = false;
	rasterizationState.rasterizerDiscardEnable = false;
	rasterizationState.polygonMode = vk::PolygonMode::eFill;
	rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
	rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
	rasterizationState.frontFace = vk::FrontFace::eClockwise;
	rasterizationState.lineWidth = 1.0f;
	rasterizationState.depthBiasEnable = false;

	// Multisample
	vk::PipelineMultisampleStateCreateInfo multisampleState{};
	multisampleState.flags = vk::PipelineMultisampleStateCreateFlags();
	multisampleState.rasterizationSamples = vk::SampleCountFlagBits::e1;
	multisampleState.rasterizationSamples = vk::SampleCountFlagBits::e1;
	multisampleState.sampleShadingEnable = false;
	
	// Color blend
	vk::PipelineColorBlendAttachmentState colorAttachment{};
	colorAttachment.blendEnable = false;
	colorAttachment.colorWriteMask = 
		vk::ColorComponentFlagBits::eR |
		vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA;

	vk::PipelineColorBlendStateCreateInfo colorBlendState{};
	colorBlendState.flags = vk::PipelineColorBlendStateCreateFlags();
	colorBlendState.logicOpEnable = false;
	colorBlendState.attachmentCount = 1;
	colorBlendState.pAttachments = &colorAttachment;

	// Pipeline
	vk::GraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.flags = vk::PipelineCreateFlags();
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStageInfos;
	pipelineInfo.pVertexInputState = &vertexInputState;
	pipelineInfo.pInputAssemblyState = &assemblyState;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizationState;
	pipelineInfo.pMultisampleState = &multisampleState;
	pipelineInfo.pColorBlendState = &colorBlendState;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;

	pipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo).value;

	device.destroyShaderModule(vertexModule);
	device.destroyShaderModule(fragmentModule);
}

/**
 * Command pool, buffers
 */
void App::createCommandPool() {
	auto indices = findQueueFamilies(physicalDevice);

	vk::CommandPoolCreateInfo commandPoolInfo{};
	commandPoolInfo.flags = vk::CommandPoolCreateFlags() |
		vk::CommandPoolCreateFlagBits::eTransient |
		vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	commandPoolInfo.queueFamilyIndex = indices.graphicsFamily.value();

	commandPool = device.createCommandPool(commandPoolInfo);
}

void App::createCommandBuffers() {
	vk::CommandBufferAllocateInfo commandBufferInfo{};
	commandBufferInfo.commandPool = commandPool;
	commandBufferInfo.level = vk::CommandBufferLevel::ePrimary;
	commandBufferInfo.commandBufferCount = swapchainData.framebuffers.size();

	commandBuffers = device.allocateCommandBuffers(commandBufferInfo);
}

void App::recordCommandBuffer(vk::CommandBuffer commandBuffer, vk::Framebuffer framebuffer) {
	vk::CommandBufferBeginInfo beginInfo{};
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	vk::RenderPassBeginInfo renderPassBegin{};
	renderPassBegin.renderPass = renderPass;
	renderPassBegin.framebuffer = framebuffer;
	renderPassBegin.renderArea.extent = swapchainData.extent;
	renderPassBegin.renderArea.offset = vk::Offset2D{0, 0};

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f);
	renderPassBegin.clearValueCount = 1;
	renderPassBegin.pClearValues = &clearColor;

	commandBuffer.begin(beginInfo);
	commandBuffer.beginRenderPass(renderPassBegin, vk::SubpassContents::eInline);

	vk::Rect2D scissor;
	scissor.extent = swapchainData.extent;
	scissor.offset = vk::Offset2D{0, 0};
	commandBuffer.setScissor(0, scissor);

	vk::Viewport viewport;
	viewport.x = 0;
	viewport.y = 0;
	viewport.width = swapchainData.extent.width;
	viewport.height = swapchainData.extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	commandBuffer.setViewport(0, viewport);


	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	commandBuffer.draw(3, 1, 0, 0);

	commandBuffer.endRenderPass();
	commandBuffer.end();
}

/**
 * Synchronization
 */
void App::createSyncObjects() {
	// Semaphores
	vk::SemaphoreCreateInfo semaphoreInfo{vk::SemaphoreCreateFlags()};

	imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
	renderFinishedSemaphore = device.createSemaphore(semaphoreInfo);

	// Fence
	vk::FenceCreateInfo fenceInfo{};
	fenceInfo.flags = vk::FenceCreateFlags() | vk::FenceCreateFlagBits::eSignaled;

	inFlightFence = device.createFence(fenceInfo);
}

void App::renderTriangle() {
	(void)device.waitForFences(inFlightFence, true, UINT32_MAX);
	device.resetFences(inFlightFence);

	// acquire image
	uint32_t imageIndex = device.acquireNextImageKHR(swapchainData.swapchain, 
			UINT32_MAX, imageAvailableSemaphore, nullptr).value;

	// record and submit command buffer
	vk::CommandBuffer commandBuffer = commandBuffers[imageIndex];
	recordCommandBuffer(commandBuffer, swapchainData.framebuffers[imageIndex]);

	vk::SubmitInfo submitInfo{};
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
	vk::PipelineStageFlags waitStages[] {vk::PipelineStageFlagBits::eColorAttachmentOutput };
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &renderFinishedSemaphore;

	graphicsQueue.submit(submitInfo, inFlightFence);

	// present image to screen
	vk::PresentInfoKHR presentInfo{};
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapchainData.swapchain;
	presentInfo.pImageIndices = &imageIndex;

	(void)presentQueue.presentKHR(presentInfo);
}

/**
 * Main loop
 */
void App::run() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		renderTriangle();
	}
}

/**
 * Entry point
 */
int main(int argc, const char **argv) {
	try {
		App app;

		app.run();

	} catch (const std::exception &err) {
		std::cerr << "[ERROR] " << err.what();
	}

	return EXIT_SUCCESS;
}
