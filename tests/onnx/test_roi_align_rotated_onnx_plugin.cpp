#include <string>
#include <vector>
#include <cstdlib>

#include "gtest/gtest.h"
#include "onnx_parser.h"
#include "parser_common.h"

using namespace ge;

class roi_align_rotated_onnx_plugin_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "roi_align_rotated_onnx_plugin_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "roi_align_rotated_onnx_plugin_test TearDown" << std::endl;
    }
};

TEST_F(roi_align_rotated_onnx_plugin_test, roi_align_rotated_onnx_plugin_test_case)
{
    CleanGlobal();
    ge::Graph graph;

    std::cout << __FILE__ << std::endl;
    std::string caseDir = __FILE__;
    std::size_t idx = caseDir.find_last_of("/");
    caseDir = caseDir.substr(0, idx);
    std::string scriptFile = caseDir + "/roi_align_rotated_plugin.py";
    std::string command = "python3 " + scriptFile;
    int ret = system(command.c_str());
    EXPECT_EQ(ret, 0);
    std::string modelFile = caseDir + "/roi_align_rotated.onnx";
    std::map<ge::AscendString, ge::AscendString> parser_params;

    auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    std::vector<ge::GNode> nodes = graph.GetAllNodes();
    EXPECT_EQ(nodes.size(), 3);
}