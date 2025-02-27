00data2jsonandsample.py 采样

01DyGenKN.py 知识点提取

02DyGenPurport.py 主旨总结

03DyGenQnet.py 联网检索

04DyGenQsetQs.py 出普通选择题

04DyGenQsetQs-Bloom.py 出Bloom选择题

05CQ_Test_Eval.py 判断LLM回答选择题是否正确

05CQ_Test_Eval_addStatic.py 通过在提示词中加入静态数据模拟污染

08countScoreCQ.py 统计LLM回答选择题的得分

21QualityControl_Q.py 对数据集进行质量检查

30ExtractDataReconstruct.py 从数据集中选取题目进行重构

31DyReconstructQs.py 将选取的题目进行重构

32MergeJson.py 将重构的数据与原来剩下没被选中的数据进行合并

DyGen.sh 执行01DyGenKN.py、02DyGenPurport.py、03DyGenQnet.py、04DyGenQsetQs.py，自动生成动态数据
