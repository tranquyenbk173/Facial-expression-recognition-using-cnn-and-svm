Chào các bạn,
Hôm nay mình sẽ giới thiệu cho các bạn về chủ đề nhận dạng cảm xúc qua khuôn mặt, sử dụng 2 phương pháp chính là SVM và CNN.
Mục tiêu của bài viết:
+ So sánh phương pháp SVM và CNN trong nhận dạng cảm xúc qua khuôn mặt.
+ So sánh phương pháp CNN cơ bản và CNN cơ bản kết hợp các đặc trưng truyền thống.

Để làm rõ 2 mục tiêu trên, mình sẽ giới thiệu mô hình kiến trúc mình triển khai (Mô hình này mình đã tìm hiểu và tham khảo bài báo: https://arxiv.org/abs/1612.02903 - Đây là bài báo uy tin, được đăng lên tạp chí nổi tiếng và tới thời điểm hiện tại có 71 bài báo khác tham chiếu + tham khảo đến nó)

Dưới đây là mô hình CNN kết hợp đặc trưng truyền thống cho hệ thống nhận dạng cảm xúc qua khuôn mặt
Đầu vào: hình ảnh có kích thước là 48x48 pixel, 
Đầu ra: trạng thái cảm xúc của khuôn mặt.
Mô hình hoạt đông :
