document.addEventListener('DOMContentLoaded', () => {
    // Chuyển đổi giao diện
    const loginModule = document.getElementById('loginModule');
    const registerModule = document.getElementById('registerModule');

    document.getElementById('linkToRegister').addEventListener('click', () => {
        loginModule.style.display = 'none';
        registerModule.style.display = 'block';
    });

    document.getElementById('linkToLogin').addEventListener('click', () => {
        registerModule.style.display = 'none';
        loginModule.style.display = 'block';
    });

    // Logic Đăng ký
    document.getElementById('btnRegister').addEventListener('click', () => {
        const user = document.getElementById('regUsername').value.trim();
        const pass = document.getElementById('regPassword').value;
        const confirmPass = document.getElementById('regConfirmPassword').value;

        if (!user || !pass) return alert("Vui lòng điền đầy đủ thông tin bác sĩ!");
        if (pass !== confirmPass) return alert("Mật khẩu xác thực chưa khớp!");
        if (localStorage.getItem('user_' + user)) return alert("Mã bác sĩ đã tồn tại!");

        localStorage.setItem('user_' + user, pass);
        alert("Chúc mừng! Tài khoản được tạo thành công, vui lòng đăng nhập!");
        document.getElementById('linkToLogin').click();
    });

    // Logic Đăng nhập
    document.getElementById('btnLogin').addEventListener('click', () => {
        const user = document.getElementById('logUsername').value.trim();
        const pass = document.getElementById('logPassword').value;

        if (!user || !pass) return alert("Bạn chưa nhập đủ thông tin!");

        const storedPass = localStorage.getItem('user_' + user);

        if (storedPass && storedPass === pass) {
            // Cấp "Thẻ hành nghề" (Token) và đuổi thẳng vào phòng 3D
            localStorage.setItem('session_token', 'valid_doctor_' + user);
            window.location.href = 'scanner.html';
        } else {
            alert("Sai tên hoặc mật khẩu!");
        }
    });
});