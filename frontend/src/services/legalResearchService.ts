import type {
  ChatRequest,
  Citation,
  Conversation,
  LegalAnswer,
} from "../types/legal";

export const promptSuggestions = [
  "Tóm tắt quy định về hợp đồng lao động",
  "Tìm án lệ liên quan đến tranh chấp đất đai",
  "So sánh quy định trong pháp điển và văn bản gốc",
  "Kiểm tra hiệu lực của quy định về xử phạt hành chính",
];

const citations: Citation[] = [
  {
    id: "vbpl-bo-luat-lao-dong-2019-d13",
    sourceType: "van-ban",
    title: "Bộ luật Lao động 2019",
    article: "Điều 13",
    clause: "Khoản 1",
    effectiveDate: "01/01/2021",
    status: "Còn hiệu lực",
    agency: "Quốc hội",
    detailUrl: "https://vbpl.vn/",
    summary:
      "Xác định hợp đồng lao động là sự thỏa thuận về việc làm có trả công, tiền lương, điều kiện lao động, quyền và nghĩa vụ của mỗi bên.",
    excerpt:
      "Khi một thỏa thuận thể hiện việc làm có trả công, tiền lương và sự quản lý, điều hành, giám sát thì được xem là hợp đồng lao động.",
    related: ["Điều 14", "Điều 20", "Điều 21"],
  },
  {
    id: "phap-dien-lao-dong-hop-dong",
    sourceType: "phap-dien",
    title: "Pháp điển chủ đề Lao động, việc làm",
    article: "Mục Hợp đồng lao động",
    effectiveDate: "Đang cập nhật theo văn bản gốc",
    status: "Theo trạng thái văn bản được pháp điển",
    agency: "Bộ Tư pháp",
    detailUrl: "https://phapdien.moj.gov.vn/",
    summary:
      "Tập hợp quy phạm về giao kết, thực hiện, sửa đổi, chấm dứt hợp đồng lao động theo cấu trúc pháp điển.",
    excerpt:
      "Các quy phạm được sắp xếp theo đề mục, thuận tiện đối chiếu với văn bản gốc và văn bản sửa đổi, bổ sung.",
    related: ["Đề mục Tiền lương", "Đề mục Kỷ luật lao động"],
  },
  {
    id: "an-le-04-2016-hop-dong",
    sourceType: "an-le",
    title: "Án lệ số 04/2016/AL",
    article: "Tình huống pháp lý",
    effectiveDate: "06/04/2016",
    status: "Được áp dụng",
    agency: "Hội đồng Thẩm phán TANDTC",
    detailUrl: "https://anle.toaan.gov.vn/",
    summary:
      "Gợi mở cách đánh giá hiệu lực và bản chất thỏa thuận trong quan hệ dân sự, có thể dùng để tham khảo phương pháp lập luận.",
    excerpt:
      "Tòa án xem xét ý chí thực sự của các bên, quá trình thực hiện và hậu quả pháp lý phát sinh từ thỏa thuận.",
    related: ["Án lệ số 09/2016/AL", "Án lệ số 42/2021/AL"],
  },
  {
    id: "vbpl-luat-dat-dai-2024-d236",
    sourceType: "van-ban",
    title: "Luật Đất đai 2024",
    article: "Điều 236",
    clause: "Khoản 2",
    effectiveDate: "01/08/2024",
    status: "Còn hiệu lực",
    agency: "Quốc hội",
    detailUrl: "https://vbpl.vn/",
    summary:
      "Quy định thẩm quyền giải quyết tranh chấp đất đai và điều kiện lựa chọn cơ quan giải quyết tùy theo giấy tờ, chứng cứ về quyền sử dụng đất.",
    excerpt:
      "Tranh chấp đã được hòa giải mà không thành có thể được giải quyết theo thủ tục hành chính hoặc tố tụng tùy trường hợp.",
    related: ["Điều 235", "Điều 237"],
  },
  {
    id: "phap-dien-dat-dai-tranh-chap",
    sourceType: "phap-dien",
    title: "Pháp điển chủ đề Đất đai",
    article: "Đề mục Giải quyết tranh chấp",
    effectiveDate: "Theo văn bản hợp nhất trong đề mục",
    status: "Theo trạng thái văn bản được pháp điển",
    agency: "Bộ Tư pháp",
    detailUrl: "https://phapdien.moj.gov.vn/",
    summary:
      "Hệ thống hóa quy phạm về hòa giải, thẩm quyền, trình tự giải quyết tranh chấp và khiếu nại đất đai.",
    excerpt:
      "Nội dung pháp điển giúp rà soát quan hệ giữa luật, nghị định và thông tư trong cùng chủ đề.",
    related: ["Đề mục Khiếu nại", "Đề mục Bồi thường, hỗ trợ, tái định cư"],
  },
  {
    id: "an-le-35-2020-dat-dai",
    sourceType: "an-le",
    title: "Án lệ số 35/2020/AL",
    article: "Về người Việt Nam định cư ở nước ngoài nhận chuyển nhượng quyền sử dụng đất",
    effectiveDate: "15/04/2020",
    status: "Được áp dụng",
    agency: "Hội đồng Thẩm phán TANDTC",
    detailUrl: "https://anle.toaan.gov.vn/",
    summary:
      "Làm rõ hướng xử lý khi giao dịch đất đai có yếu tố người Việt Nam định cư ở nước ngoài và người đứng tên hộ.",
    excerpt:
      "Tòa án xem xét nguồn tiền, ý chí giao dịch và quan hệ giữa người đứng tên với người có quyền lợi thực tế.",
    related: ["Án lệ số 02/2016/AL", "Án lệ số 14/2017/AL"],
  },
  {
    id: "vbpl-xu-phat-hanh-chinh-2012-d3",
    sourceType: "van-ban",
    title: "Luật Xử lý vi phạm hành chính 2012, sửa đổi 2020",
    article: "Điều 3",
    effectiveDate: "01/01/2022",
    status: "Còn hiệu lực một phần theo sửa đổi",
    agency: "Quốc hội",
    detailUrl: "https://vbpl.vn/",
    summary:
      "Nêu nguyên tắc xử lý vi phạm hành chính, bao gồm tính kịp thời, công khai, khách quan, đúng thẩm quyền và đúng quy định.",
    excerpt:
      "Mọi vi phạm hành chính phải được phát hiện, ngăn chặn kịp thời và xử lý nghiêm minh theo quy định.",
    related: ["Điều 4", "Điều 23", "Điều 56"],
  },
];

export const seedConversations: Conversation[] = [
  {
    id: "conv-lao-dong",
    title: "Hợp đồng lao động",
    category: "Lao động",
    updatedAt: "2026-06-08T08:45:00.000Z",
    lastSummary: "Tóm tắt căn cứ về hình thức và nội dung hợp đồng.",
    messages: [
      {
        id: "msg-seed-user",
        role: "user",
        content: "Tóm tắt quy định về hợp đồng lao động",
        status: "complete",
        createdAt: "2026-06-08T08:44:00.000Z",
      },
      {
        id: "msg-seed-assistant",
        role: "assistant",
        content:
          "Hợp đồng lao động được xác định theo bản chất quan hệ: có việc làm, trả công hoặc tiền lương, và có sự quản lý, điều hành, giám sát của một bên. Khi rà soát, nên đối chiếu văn bản gốc để kiểm tra điều kiện giao kết, hình thức hợp đồng, loại hợp đồng và các nội dung bắt buộc. Pháp điển giúp nhìn nhanh cấu trúc quy phạm theo chủ đề, còn án lệ chỉ nên dùng để tham khảo cách lập luận trong tình huống tương tự.",
        status: "complete",
        createdAt: "2026-06-08T08:45:00.000Z",
        citations: citations.slice(0, 3),
        confidence: "cao",
        followUps: [
          "Liệt kê nội dung bắt buộc của hợp đồng lao động",
          "So sánh hợp đồng lao động và hợp đồng dịch vụ",
        ],
      },
    ],
  },
  {
    id: "conv-dat-dai",
    title: "Tranh chấp đất đai",
    category: "Đất đai",
    updatedAt: "2026-06-07T15:30:00.000Z",
    lastSummary: "Tìm án lệ và quy định về thẩm quyền giải quyết.",
    messages: [],
  },
  {
    id: "conv-hanh-chinh",
    title: "Xử phạt hành chính",
    category: "Hành chính",
    updatedAt: "2026-06-06T10:05:00.000Z",
    lastSummary: "Kiểm tra hiệu lực và nguyên tắc xử phạt.",
    messages: [],
  },
];

const delay = (ms: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, ms));

const normalize = (value: string) =>
  value
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();

const byIds = (ids: string[]) =>
  ids
    .map((id) => citations.find((citation) => citation.id === id))
    .filter((citation): citation is Citation => Boolean(citation));

export async function searchLegalAnswer(
  request: ChatRequest
): Promise<LegalAnswer> {
  await delay(760);

  const normalizedQuestion = normalize(request.question);

  if (
    normalizedQuestion.includes("loi") ||
    normalizedQuestion.includes("mat ket noi")
  ) {
    throw new Error("Không thể kết nối dịch vụ tra cứu.");
  }

  if (
    normalizedQuestion.includes("khong tim thay") ||
    normalizedQuestion.includes("xyz") ||
    normalizedQuestion.includes("khong co ket qua")
  ) {
    return {
      answer:
        "Chưa tìm thấy căn cứ đủ sát với câu hỏi. Bạn có thể bổ sung lĩnh vực pháp luật, mốc thời gian, cơ quan ban hành hoặc tình huống tranh chấp cụ thể để thu hẹp phạm vi tra cứu.",
      citations: [],
      confidence: "thấp",
      followUps: [
        "Tìm theo số hiệu văn bản",
        "Tìm theo điều khoản và ngày hiệu lực",
      ],
      noResultReason: "Không có nguồn phù hợp trong dữ liệu mẫu.",
    };
  }

  if (
    normalizedQuestion.includes("dat dai") ||
    normalizedQuestion.includes("tranh chap dat") ||
    normalizedQuestion.includes("an le")
  ) {
    return {
      answer:
        "Với tranh chấp đất đai, cần tách ba lớp căn cứ. Thứ nhất là văn bản pháp luật để xác định thẩm quyền, điều kiện hòa giải và thủ tục giải quyết. Thứ hai là pháp điển để rà nhanh các quy phạm cùng đề mục và văn bản hướng dẫn liên quan. Thứ ba là án lệ để tham khảo cách tòa án đánh giá chứng cứ, ý chí giao dịch và người có quyền lợi thực tế. Khi áp dụng, nên ưu tiên văn bản còn hiệu lực tại thời điểm phát sinh tranh chấp.",
      citations: byIds([
        "vbpl-luat-dat-dai-2024-d236",
        "phap-dien-dat-dai-tranh-chap",
        "an-le-35-2020-dat-dai",
      ]),
      confidence: "cao",
      followUps: [
        "Tóm tắt thủ tục hòa giải tranh chấp đất đai",
        "Tìm án lệ về người đứng tên hộ quyền sử dụng đất",
      ],
    };
  }

  if (
    normalizedQuestion.includes("so sanh") ||
    normalizedQuestion.includes("phap dien") ||
    normalizedQuestion.includes("van ban goc")
  ) {
    return {
      answer:
        "Văn bản gốc là căn cứ pháp lý trực tiếp để xác định quyền, nghĩa vụ và hiệu lực áp dụng. Pháp điển không thay thế văn bản gốc, nhưng giúp hệ thống hóa quy phạm theo chủ đề, thuận tiện kiểm tra quan hệ giữa luật, nghị định và thông tư. Án lệ không phải văn bản quy phạm pháp luật, song có giá trị định hướng áp dụng pháp luật trong tình huống tương tự. Cách tra cứu chắc chắn là đi từ pháp điển để khoanh vùng, mở văn bản gốc để kiểm tra hiệu lực, rồi đối chiếu án lệ nếu có tranh chấp cụ thể.",
      citations: byIds([
        "phap-dien-lao-dong-hop-dong",
        "vbpl-bo-luat-lao-dong-2019-d13",
        "an-le-04-2016-hop-dong",
      ]),
      confidence: "trung bình",
      followUps: [
        "Kiểm tra hiệu lực văn bản gốc",
        "Liệt kê án lệ có tình huống tương tự",
      ],
    };
  }

  if (
    normalizedQuestion.includes("xu phat") ||
    normalizedQuestion.includes("hanh chinh") ||
    normalizedQuestion.includes("hieu luc")
  ) {
    return {
      answer:
        "Khi kiểm tra quy định xử phạt hành chính, cần xác định văn bản đang có hiệu lực tại thời điểm hành vi xảy ra, thẩm quyền xử phạt, thời hiệu, tình tiết tăng nặng hoặc giảm nhẹ và hình thức xử phạt bổ sung nếu có. Nếu quy định đã được sửa đổi, cần xem phần chuyển tiếp và văn bản hợp nhất để tránh áp dụng nhầm.",
      citations: byIds(["vbpl-xu-phat-hanh-chinh-2012-d3"]),
      confidence: "trung bình",
      followUps: [
        "Kiểm tra thẩm quyền lập biên bản",
        "Tìm nguyên tắc áp dụng văn bản có lợi hơn",
      ],
    };
  }

  return {
    answer:
      "Có thể bắt đầu bằng việc xác định loại nguồn cần dùng: văn bản pháp luật để tìm quy phạm đang áp dụng, pháp điển để rà cấu trúc quy phạm theo chủ đề, và án lệ để tham khảo cách giải quyết tình huống tranh chấp. Với câu hỏi này, tôi đề xuất mở văn bản gốc trước, sau đó đối chiếu pháp điển và án lệ nếu có dấu hiệu tranh chấp hoặc cách hiểu khác nhau.",
    citations: byIds([
      "vbpl-bo-luat-lao-dong-2019-d13",
      "phap-dien-lao-dong-hop-dong",
      "an-le-04-2016-hop-dong",
    ]),
    confidence: "trung bình",
    followUps: [
      "Thu hẹp theo lĩnh vực pháp luật",
      "Tìm theo số hiệu hoặc điều khoản",
    ],
  };
}
