### Code completion

#### Tiny Starcoder

In my approach, I used the Tiny Starcoder model for code completion, evaluating its performance on a test dataset generated from my past project files.

Initially, I tested the model by splitting the code into individual lines, omitting one line at a time, and comparing the generated output with the original line. The model’s performance on this task showed limited effectiveness, as reflected by CHRF and ROUGE scores, both yielding an accuracy of about 27%, indicating potential for improvement. Positively, the model achieved over 60% accuracy in syntax, meaning it could generally maintain correct syntax structure. However, only 37% of examples were fully completed as intended.

In the next example, I simulate a scenario where the cursor is placed in the middle of a line, and the model is expected to complete the line with generated code. The results show a slight decrease in performance compared to the previous approach—only 21% of examples are completed correctly. In the prior setup, where the model generated entire lines of code, the accuracy was higher.


For higher performance, models like the full-size Starcoder or CodeLlama would be preferable, though they are too computationally demanding to run effectively on my current setup.



#### Metrics

I employed several metrics to assess the model’s performance, including CHRF, exact match, syntax validation, ROUGE-L, and my own hand-labeling.

- CHRF: measures character-level similarity based on the longest shared subsequence of characters.

- Syntax Check: I used Python’s compile() function to determine if the generated code is syntactically correct.

- ROUGE-L: Evaluates word-level similarity by calculating the longest common subsequence between generated and reference code.


#### Dataset 

The code files were sourced from my personal machine learning project repository, offering a range of relevant code examples to evaluate the model’s performance across different scenarios and coding patterns.