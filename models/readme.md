trained models are saved here

模型的测试命令的参数设置可以参考如下命令（ADNet和psftd-shar使用了选项`--ea`，ahdr没有使用）

- kal-adnet

  ```shell
  python test.py --model-type adnet --checkpoint /.../hdr-toolkit/models/kal-adnet/ --data kalantari --data-with-gt --input-dir /.../hdr-toolkit/samples --output-dir /.../hdr-toolkit/test_result/ --out-activation sigmoid --ea
  ```

- kal-ahdr

  ```shell
  python test.py --model-type ahdr --checkpoint /.../hdr-toolkit/models/kal-ahdr/ --data kalantari --data-with-gt --input-dir /.../hdr-toolkit/samples --output-dir /.../hdr-toolkit/test_result/ --out-activation sigmoid
  ```

- psftd-share

  ```shell
  python test.py --model-type psftd-share --checkpoint /.../hdr-toolkit/models/psftd-share/ --data kalantari --data-with-gt --input-dir /.../hdr-toolkit/samples --output-dir /.../hdr-toolkit/test_result/ --out-activation sigmoid --ea
  ```

  